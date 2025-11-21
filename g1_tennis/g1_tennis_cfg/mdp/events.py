# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import carb
from isaaclab.markers.visualization_markers import VisualizationMarkers
import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def reset_hit(
    env: ManagerBasedRLEnv, 
    env_ids: torch.Tensor,
    pos_range_x: tuple[float, float],
    pos_range_y: tuple[float, float], 
    pos_range_z: tuple[float, float],    
):
    """Reset the hit position for the specified environments.

    Args:
        env_ids: List of environment ids which must be reset.
    """
    # limit of hit position relative to the robot base

    env.x_min, env.x_max = pos_range_x
    env.y_min, env.y_max = pos_range_y
    env.z_min, env.z_max = pos_range_z

    # randomly sample the hit position within the specified limits
    hit_x = torch.rand(len(env_ids), device=env.device) * (pos_range_x[1] - pos_range_x[0]) + pos_range_x[0]
    hit_y = torch.rand(len(env_ids), device=env.device) * (pos_range_y[1] - pos_range_y[0]) + pos_range_y[0]
    hit_z = torch.rand(len(env_ids), device=env.device) * (pos_range_z[1] - pos_range_z[0]) + pos_range_z[0]

    # reset the hit position for the specified environments
    env.hit_position[env_ids] = torch.stack((hit_x, hit_y, hit_z), dim=-1)
    env.hit_position_w[env_ids] = env.scene["robot"].data.root_pos_w[env_ids].clone() + env.hit_position[env_ids]

    # hit time randomization, percentage of the episode length
    percentage_min = 0.4
    percentage_max = 0.6
    # randomly sample the hit time within the specified limits
    hit_time_percentage = torch.rand(len(env_ids), device=env.device) * (percentage_max - percentage_min) + percentage_min
    # calculate the target intercept time based on the episode length
    # episode length and then by the percentage
    env.target_intercept_time[env_ids] = env.max_episode_length * hit_time_percentage.view(-1, 1)* 0.02    

def reset_ball(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    initial_vel: float = -10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """将球重置到 hit_pos_w 位置"""
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 克隆默认状态
    root_state = asset.data.default_root_state[env_ids].clone()
    
    # 修改位置为 hit_pos_w（保持默认的姿态和速度）
    root_state[:, :3] = env.hit_position_w[env_ids]
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)  # 默认四元数
    root_state[:, 7:] = 0.0  # 速度清零

    displacement_x = -initial_vel * env.target_intercept_time[env_ids].squeeze(-1)  # 如果v_x=-12, t=1, 则位移=+12
    initial_x = env.hit_position_w[:, 0] + displacement_x
    root_state[:, 0] = initial_x

    gravity = abs(env.sim.cfg.gravity[2])
    target_time = env.target_intercept_time[env_ids].squeeze(-1)
    v_x = initial_vel
    v_z = gravity * target_time / 2.0
    root_state[:, 7] = v_x
    root_state[:, 9] = v_z

    # 分别写入 pose 和 velocity
    asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)      # 位置 + 四元数
    asset.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)
    # 重置内部缓冲区
    asset.reset(env_ids)

def compute_ball_trajectories(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor
):
    """
    为指定的环境计算并存储本回合的拦截目标（时间和位置）。
    该函数在 event_manager 重置了场景状态后被调用。
    """
    if len(env_ids) == 0:
        return

    num_resets = len(env_ids)

    # 1. 从场景中读取机器人和球的初始状态
    robot_pos = env.scene["robot"].data.root_pos_w[env_ids]
    ball_initial_pos = env.scene["object"].data.root_pos_w[env_ids]
    full_velocity = env.scene["object"].data.root_vel_w[env_ids]
    ball_initial_vel = full_velocity[:, 0:3]

    # 2. 定义机器人和拦截平面参数
    robot_forward_vec = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(num_resets, -1)
    robot_up_vec = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(num_resets, -1)
    robot_right_vec = torch.tensor([0.0, -1.0, 0.0], device=env.device).expand(num_resets, -1)

    gravity = torch.tensor([0.0, 0.0, -9.81], device=env.device)

    plane_dist = 0.3
    reachable_rect = {'y_min': -0.5, 'y_max': 0.5, 'z_min': -0.2, 'z_max': 1.0}

    # 3. 向量化求解拦截时刻 T_intercept
    N_plane = robot_forward_vec
    P_plane = robot_pos + plane_dist * N_plane

    # 系数 A, B, C
    A = 0.5 * torch.einsum("bi,i->b", N_plane, gravity)
    B = torch.einsum("bi,bi->b", ball_initial_vel, N_plane)
    C = torch.einsum("bi,bi->b", ball_initial_pos - P_plane, N_plane)

    # 【核心修正区域开始】

    # 初始化目标时间为无穷大，表示默认无解。这比用超时时间更清晰。
    intercept_times = torch.full((num_resets,), float('inf'), device=env.device)

    # 定义一个阈值来判断 A 是否接近于 0
    linear_mask = torch.abs(A) < 1e-6
    quadratic_mask = ~linear_mask

    # --- 情况1：A 接近于 0，作为一元一次方程求解 ---
    if torch.any(linear_mask):
        # 提取出对应的数据
        B_lin = B[linear_mask]
        C_lin = C[linear_mask]
        
        # 创建一个子集张量，用于存放该分支的解
        solutions_lin = torch.full_like(B_lin, float('inf'))
        
        # 避免除以0 (如果B也为0，表示球与平面相对静止，无解)
        b_not_zero_mask = torch.abs(B_lin) > 1e-6
        
        if torch.any(b_not_zero_mask):
            B_valid = B_lin[b_not_zero_mask]
            C_valid = C_lin[b_not_zero_mask]
            
            # 计算线性解 t = -C / B
            t_linear = -C_valid / B_valid
            
            # 过滤掉过去或当前的时间点
            t_linear[t_linear <= 1e-4] = float('inf')
            
            # 将有效解填入子集张量
            solutions_lin[b_not_zero_mask] = t_linear
        
        # 将此分支的解更新到最终结果张量中
        intercept_times[linear_mask] = solutions_lin

    # --- 情况2：A 不为 0，作为一元二次方程求解 ---
    if torch.any(quadratic_mask):
        # 提取出对应的数据
        A_quad = A[quadratic_mask]
        B_quad = B[quadratic_mask]
        C_quad = C[quadratic_mask]

        # 创建一个子集张量，用于存放该分支的解
        solutions_quad = torch.full_like(A_quad, float('inf'))
        
        discriminant = B_quad**2 - 4 * A_quad * C_quad
        
        # 只处理有实数解的环境
        valid_discriminant_mask = discriminant >= 0
        if torch.any(valid_discriminant_mask):
            # 再次过滤，只对有实数解的子集进行计算
            A_valid = A_quad[valid_discriminant_mask]
            B_valid = B_quad[valid_discriminant_mask]
            sqrt_discriminant = torch.sqrt(discriminant[valid_discriminant_mask])

            t1 = (-B_valid - sqrt_discriminant) / (2 * A_valid + 1e-8)
            t2 = (-B_valid + sqrt_discriminant) / (2 * A_valid + 1e-8)

            t1[t1 <= 1e-4] = float('inf')
            t2[t2 <= 1e-4] = float('inf')
            
            final_solutions = torch.min(t1, t2)

            # 将有效解填入子集张量
            solutions_quad[valid_discriminant_mask] = final_solutions

        # 将此分支的解更新到最终结果张量中
        intercept_times[quadratic_mask] = solutions_quad

    # 【核心修正区域结束】

    # 4. 计算所有潜在的拦截点（即使时间是inf）
    # 注意：这里需要处理 t=inf 的情况，否则计算会出错
    # 我们只为有有效时间的点计算位置
    valid_time_mask = intercept_times < float('inf')
    t = intercept_times.unsqueeze(1)
    # 初始化拦截点位置为机器人当前位置（或一个安全默认值）
    intercept_pos_world = robot_pos.clone() 

    # 只对有解的环境计算拦截点
    if torch.any(valid_time_mask):
        intercept_pos_world[valid_time_mask] = ball_initial_pos[valid_time_mask] + \
                                                ball_initial_vel[valid_time_mask] * t[valid_time_mask] + \
                                                0.5 * gravity * t[valid_time_mask]**2

    # 5. 向量化可行性检查
    relative_pos = intercept_pos_world - robot_pos
    y_local = torch.einsum("bi,bi->b", relative_pos, robot_right_vec)
    z_local = torch.einsum("bi,bi->b", relative_pos, robot_up_vec)

    y_ok = (y_local >= reachable_rect['y_min']) & (y_local <= reachable_rect['y_max'])
    z_ok = (z_local >= reachable_rect['z_min']) & (z_local <= reachable_rect['z_max'])
    reachable_mask = y_ok & z_ok

    # 6. 最终确定有效的环境
    # 一个目标是有效的，必须同时满足：时间有解(valid_time_mask) 且 位置可达(reachable_mask)
    final_valid_mask = valid_time_mask & reachable_mask

    # 7. 将计算出的、且可行的目标存储到环境的状态变量中
    #    首先，将所有需要重置的环境的目标设为默认失败值
    env.target_intercept_time[env_ids] = env.max_episode_length * 0.02
    env.target_intercept_pos[env_ids] = env.scene["robot"].data.default_root_state[env_ids, :3]

    #    然后，只为那些最终有效的环境，填入正确的拦截目标
    valid_env_ids = env_ids[final_valid_mask]
    if len(valid_env_ids) > 0:
        env.target_intercept_time[valid_env_ids] = intercept_times[final_valid_mask]
        env.target_intercept_pos[valid_env_ids] = intercept_pos_world[final_valid_mask]