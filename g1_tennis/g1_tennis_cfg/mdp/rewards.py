# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_close_to_ball(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    object_ball: RigidObject = env.scene[object_cfg.name]
    # calculate the relative position
    relative_pos = object_ball.data.root_pos_w - robot.data.root_pos_w
    # calculate the distance
    distance = torch.norm(relative_pos, dim=-1)
    # calculate the reward based on the distance, the closer the better
    reward = torch.exp(-distance / 0.05)  # Adjust the std as needed
    return reward

def racket_close_to_ball(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot",
                                                                    body_names=["right_wrist_yaw_link"]), 
                                                                    object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Reward the racket for being close to the ball."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object_ball: RigidObject = env.scene[object_cfg.name]
    relative_pos = robot.data.body_link_pos_w[:, 29] - object_ball.data.root_pos_w
    # calculate the distance
    distance = torch.norm(relative_pos, dim=-1)
    # calculate the reward based on the distance, the closer the better
    reward = torch.exp(-distance / 0.05)  # Adjust the std as needed
    return reward


def track_swing_position_at_impact(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    robot: RigidObject = env.scene["robot"]

    # 球拍位置（世界坐标系）
    current_pos = robot.data.body_link_pos_w[:, 37]  # [num_envs, 3]

    # 将目标位置从 base 坐标系变换到世界坐标系
    base_pos = robot.data.root_pos_w  # [num_envs, 3]
    base_quat = robot.data.root_quat_w  # [num_envs, 4]
    # target_pos_local = env.hit_position  # [num_envs, 3]
    target_pos = env.hit_position_w
    # target_pos = math_utils.quat_apply(base_quat, target_pos_local) + base_pos  # 世界坐标下的目标位置

    # 时间门控
    time_remaining = env.target_intercept_time.view(-1) - env.episode_length_buf * 0.02
    impact_mask = (time_remaining < 0.01) & (time_remaining >= -0.01)
    if torch.any(impact_mask) & env.debug_mode == True:
        distance = torch.norm(current_pos[impact_mask] - target_pos[impact_mask], dim=1)
        print("### distance at impact: ", distance)

    # 奖励计算
    pos_error_sq = torch.sum(torch.square(current_pos - target_pos), dim=1)
    reward = 1.0 / (1.0 + pos_error_sq / (std ** 2))
    # print('## pos time remaining:',time_remaining)

    final_reward = torch.zeros_like(reward)
    final_reward[impact_mask] = reward[impact_mask]
    return final_reward.view(env.num_envs)


def track_swing_position_dense(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    robot: RigidObject = env.scene["robot"]

    current_pos = robot.data.body_link_pos_w[:, 37]
    base_pos = robot.data.root_pos_w
    base_quat = robot.data.root_quat_w
    target_pos_local = env.hit_position_w

    target_pos = math_utils.quat_apply(base_quat, target_pos_local) + base_pos

    pos_error_sq = torch.sum(torch.square(current_pos - target_pos), dim=1)
    reward = 1.0 / (1.0 + pos_error_sq / (std ** 2))
    return reward.view(env.num_envs)


def track_swing_velocity_at_impact(env: ManagerBasedRLEnv, std: float, time_window: float = 0.02) -> torch.Tensor:
    robot: RigidObject = env.scene["robot"]
    current_vel = robot.data.body_link_vel_w[:, 37, :3]  # 世界速度

    # 定义目标方向（base 坐标系前方）
    base_quat = robot.data.root_quat_w  # [num_envs, 4]
    target_direction_base = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1)
    target_direction = math_utils.quat_apply(base_quat, target_direction_base)  # 旋转到世界坐标系

    target_speed = 6.0
    target_vel = target_direction * target_speed  # 世界速度目标

    # 时间门控
    time_remaining = env.target_intercept_time.view(-1) - env.episode_length_buf * 0.02
    impact_mask = (time_remaining < time_window) & (time_remaining > -time_window)

    if torch.any(impact_mask) & env.debug_mode == True:
        # print("### time remaining at impact: ", time_remaining[impact_mask])
        print("### current_vel: ", current_vel[impact_mask])

    vel_error_sq = torch.sum(torch.square(current_vel - target_vel), dim=1)
    reward = 1.0 / (1.0 + vel_error_sq / (std ** 2))

    final_reward = torch.zeros_like(reward)
    final_reward[impact_mask] = reward[impact_mask]
    return final_reward.view(env.num_envs)

def track_swing_velocity_direction_at_impact(env: ManagerBasedRLEnv, std: float = 0.1) -> torch.Tensor:
    robot: RigidObject = env.scene["robot"]
    current_vel = robot.data.body_link_vel_w[:, 37, :3]

    base_quat = robot.data.root_quat_w
    target_dir_base = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1)
    target_direction = math_utils.quat_apply(base_quat, target_dir_base)

    current_vel_norm = torch.norm(current_vel, dim=1, keepdim=True) + 1e-6
    current_dir = current_vel / current_vel_norm

    cosine_sim = torch.sum(current_dir * target_direction, dim=1)  # [-1, 1]
    cosine_error_sq = (1.0 - cosine_sim) ** 2

    # 时间门控
    time_remaining = env.target_intercept_time.view(-1) - env.episode_length_buf * 0.02
    impact_mask = (time_remaining < 0.02) & (time_remaining > -0.02)

    reward = 1.0 / (1.0 + cosine_error_sq / (std ** 2))
    final_reward = torch.zeros_like(reward)
    final_reward[impact_mask] = reward[impact_mask]
    return final_reward

def track_swing_velocity_magnitude_at_impact(env: ManagerBasedRLEnv, std: float = 0.5, target_speed: float = 4.0) -> torch.Tensor:
    robot: RigidObject = env.scene["robot"]
    current_vel = robot.data.body_link_vel_w[:, 37, :3]

    current_speed = torch.norm(current_vel, dim=1)
    speed_error_sq = (current_speed - target_speed) ** 2

    # 时间门控
    time_remaining = env.target_intercept_time.view(-1) - env.episode_length_buf * 0.02
    impact_mask = (time_remaining < 0.02) & (time_remaining > -0.02)

    reward = 1.0 / (1.0 + speed_error_sq / (std ** 2))
    final_reward = torch.zeros_like(reward)
    final_reward[impact_mask] = reward[impact_mask]
    return final_reward



def track_swing_orientation_at_impact(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """
    At the moment of impact, rewards the alignment of the racket's orientation
    with a target orientation, based on the paper's formula (S4).
    """
    # --- 1. Get Actual Racket Orientation Normal Vector ---
    robot: RigidObject = env.scene["robot"]
    racket_quat_w = robot.data.body_link_quat_w[:, 37]  # [num_envs, 4]

    # b = torch.tensor([0.0, -1.0, 0.0], device=env.device)
    # a = math_utils.quat_apply(racket_quat_w, b)
    # print("拍面法向在世界坐标中的方向:", a[0])

    # for axis in [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]:
    #     local_axis = torch.tensor(axis, device=env.device).expand(env.num_envs, -1)
    #     world_axis = math_utils.quat_apply(racket_quat_w, local_axis)
    #     print(f"local {axis} → world direction:", world_axis[0])

    # Define the racket's normal vector in its local frame.
    racket_local_normal = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, -1)
    # racket_local_normal = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1)
    
    # Rotate the local normal to the world frame to get the actual facing direction
    actual_normal = math_utils.quat_apply(racket_quat_w, racket_local_normal)

    # --- 2. Define Target Racket Orientation Normal Vector ---
    # The target remains the same: have the racket face forward from the robot's base.
    base_quat = robot.data.root_quat_w  # [num_envs, 4]
    target_normal_base = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1)
    target_normal_base = target_normal_base / torch.norm(target_normal_base, dim=-1, keepdim=True) # Normalize
    target_normal = math_utils.quat_apply(base_quat, target_normal_base)

    # --- 3. Calculate Squared Cosine Distance ---
    # This logic is unchanged.
    cosine_sim = torch.sum(actual_normal * target_normal, dim=1)
    cosine_dist = 1.0 - cosine_sim
    orientation_error_sq = torch.square(cosine_dist)

    # --- 4. Calculate Reward ---
    # This logic is unchanged.
    reward = 1.0 / (1.0 + orientation_error_sq / (std ** 2))

    # --- 5. Apply Time Gating ---
    # This logic is unchanged.
    time_remaining = env.target_intercept_time.view(-1) - env.episode_length_buf * 0.02
    impact_mask = (time_remaining < 0.02) & (time_remaining > -0.02)
    
    # print('## cos time remaining:',time_remaining)
    if torch.any(impact_mask)& env.debug_mode == True:
        print("ccosine similarity at impact: ", cosine_sim[impact_mask])

    final_reward = torch.zeros_like(reward)
    final_reward[impact_mask] = reward[impact_mask]
    
    return final_reward.view(env.num_envs)


def maintain_default_pose_before_impact(env: ManagerBasedRLEnv,
                                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                        pre_impact_margin: float = 0.3) -> torch.Tensor:
    """
    Encourage the robot to stay near its default pose far before the impact.
    As impact approaches, this penalty decays smoothly to allow preparation motion.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 当前关节角度与默认姿态的偏差
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    # 距离撞击时间（正值，负值表示已经击球）
    time_remaining = env.target_intercept_time.view(-1) - env.episode_length_buf * 0.02

    # 惩罚系数：击球远时为 1，靠近时逐渐变为 0，击球后恒为 0
    decay = torch.clamp(time_remaining, min=0.2, max=1.5)

    # 线性衰减也可换成平方等非线性形式
    penalty = decay * torch.sum(angle ** 2, dim=1)
    return penalty



def maintain_default_pose_after_impact(env: ManagerBasedRLEnv,
                                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                       post_impact_delay: float = 0.3) -> torch.Tensor:
    """
    Encourage the robot to maintain its default pose after the swing impact,
    by applying a smooth penalty on joint deviations that increases over time.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    
    # time since impact
    time_remaining = env.target_intercept_time.view(-1) - env.episode_length_buf * 0.02
    time_since_impact = -time_remaining  # positive after impact

    # decay factor: 0 before impact, increases to 1 after post_impact_delay
    decay = torch.clamp((time_since_impact - post_impact_delay), min=0.0, max=1.0)
    
    # smooth penalty
    penalty = decay * torch.sum(angle ** 2, dim=1)
    return penalty



def penalize_racket_out_of_bounds(env: ManagerBasedRLEnv) -> torch.Tensor:
    x_min, x_max = -0.2, 0.1
    y_min, y_max = -0.5, -0.3
    z_min, z_max = 0.0, 0.3
    penalty = 1.0

    robot: RigidObject = env.scene["robot"]
    racket_pos_w = robot.data.body_link_pos_w[:, 37]
    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w

    vec_base_to_racket_w = racket_pos_w - base_pos_w
    racket_pos_b = math_utils.quat_apply_inverse(base_quat_w, vec_base_to_racket_w)

    out_of_bounds_x = (racket_pos_b[:, 0] < x_min) | (racket_pos_b[:, 0] > x_max)
    out_of_bounds_y = (racket_pos_b[:, 1] < y_min) | (racket_pos_b[:, 1] > y_max)
    out_of_bounds_z = (racket_pos_b[:, 2] < z_min) | (racket_pos_b[:, 2] > z_max)
    is_out_of_bounds = out_of_bounds_x | out_of_bounds_y | out_of_bounds_z

    # --- 时间门控： > 0 或 < -0.5 ---
    time_remaining = env.target_intercept_time.view(-1) - env.episode_length_buf * 0.02
    active_mask = (time_remaining > 0.2) | (time_remaining < -0.5)

    final_mask = is_out_of_bounds & active_mask

    reward = torch.zeros(env.num_envs, device=env.device)
    reward[final_mask] = penalty

    return reward

def power_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize power consumption of the robot."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.applied_torque) * torch.abs(asset.data.joint_vel), dim=1)

def homiecmd_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the robot's velocity commands using L2 squared kernel.
    
    This term targets the action slice [0:4], which corresponds to [vx, vy, vyaw, base_height].
    """
    # 获取当前和上一时间步的4维指令动作
    # action shape: (num_envs, total_action_dim)
    current_cmd = env.action_manager.action[:, 0:4]
    prev_cmd = env.action_manager.prev_action[:, 0:4]
    
    # 计算差值的L2范数的平方
    # The output shape will be (num_envs,) which is what the reward manager expects.
    return torch.sum(torch.square(current_cmd - prev_cmd), dim=1)

def walk_reward_w(
    env: ManagerBasedRLEnv,
    near_threshold: float = 0.04,
    target_speed: float = 1.0,
    pos_err_scale: float = 2.0,
    vel_err_scale: float = 2.0,
    standing_point_offset: float = 0.2
) -> torch.Tensor:
    robot: RigidObject = env.scene["robot"]
    obj: RigidObject = env.scene["object"]
    robot_quat_w = robot.data.root_quat_w       # (num_envs, 4)
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w
    obj_pos_w = obj.data.root_pos_w
    obj_quat_w = obj.data.root_quat_w
    
    # calculate buffers

    # 1. 所有可选的 box 尺寸 (来自 assets_cfg)
    box_size = torch.tensor(obj.cfg.spawn.assets_cfg[0].size,device=env.device).repeat(env.num_envs, 1)  

    env.box_size = box_size
    offset= -env.box_size[:,0] / 2 - standing_point_offset
    standing_offset_local = torch.stack([
        offset,  # x 方向，负半边长 + 额外偏移
        torch.zeros_like(offset),  # y 方向
        torch.zeros_like(offset)   # z 方向
    ], dim=-1)
    standing_offset_world = math_utils.quat_apply(obj_quat_w, standing_offset_local)
    standing_point_world = obj_pos_w + standing_offset_world
    env.standing_point_world = standing_point_world
    box_standing_points_pos = standing_point_world[..., 0:2]
    root_pos = robot_pos_w.unsqueeze(1)

    box_pos_diff = box_standing_points_pos.unsqueeze(1) - root_pos[..., 0:2]
    box_pos_err = torch.sum(box_pos_diff ** 2, dim=-1)        
    near_mask = box_pos_err <= near_threshold
    env.near_mask = near_mask
    # ---- position reward ----
    box_pos_reward = torch.exp(-pos_err_scale * box_pos_err)

    # ---- velocity reward ----
    root_vel = robot.data.root_lin_vel_w.unsqueeze(1)
    root_vel = root_vel[:, :,:2]  # only use the xy components
    # print("root_vel", root_vel)
    box_dir = torch.nn.functional.normalize(box_pos_diff, dim=-1, eps=1e-6)
    box_dir_speed = torch.sum(box_dir * root_vel, dim=-1)
    # print("box_dir_speed", box_dir_speed)
    box_vel_err = target_speed - box_dir_speed
    vel_reward = torch.exp(-vel_err_scale * box_vel_err ** 2)
    vel_reward[box_dir_speed <= 0] = 0

    # ---- facing reward ----
    root_rot = robot_quat_w.unsqueeze(1)
    heading_rot = math_utils.calc_heading_quat(root_rot)
    facing_dir = torch.zeros((env.num_envs, 1, 3), device=env.device)
    facing_dir[..., 0] = 1.0
    facing_dir = math_utils.quat_rotate(heading_rot, facing_dir)
    facing_err = torch.sum(box_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)

    # ---- mask when close ----
    box_pos_reward[near_mask] = 1.0
    vel_reward[near_mask] = 1.0
    facing_reward[near_mask] = 1.0
    return (box_pos_reward + vel_reward + facing_reward).view(-1)

def walk_position_reward_w(
    env: ManagerBasedRLEnv,
    near_threshold: float = 0.04,
    pos_err_scale: float = 2.0,
    standing_point_offset: float = 0.1
) -> torch.Tensor:
    robot: RigidObject = env.scene["robot"]
    
    robot_pos_w = robot.data.root_pos_w         # 世界坐标
    target_point_world = env.hit_position_w     # 世界坐标 (num_envs, 3)
    
    # 由于没有物体朝向，直接在世界坐标系中应用 offset
    # 这里沿世界 x 轴负方向偏移（模仿原始逻辑中物体局部 x 轴负方向）
    standing_offset_world = torch.zeros_like(target_point_world)
    standing_offset_world[..., 0] = -standing_point_offset  # x 方向偏移
    
    # 计算实际的目标站立点 = 目标点 + offset
    standing_point_world = target_point_world + standing_offset_world

    target_points_pos = standing_point_world[..., 0:2]  # 只取 xy 坐标
    root_pos = robot_pos_w.unsqueeze(1)

    target_pos_diff = target_points_pos.unsqueeze(1) - root_pos[..., 0:2]
    target_pos_err = torch.sum(target_pos_diff ** 2, dim=-1)        
    near_mask = target_pos_err <= near_threshold
    
    # ---- position reward ----
    pos_reward = torch.exp(-pos_err_scale * target_pos_err)
    
    # ---- mask when close ----
    pos_reward[near_mask] = 1.0
    
    return pos_reward.view(-1)


def walk_velocity_reward_w(
    env: ManagerBasedRLEnv,
    near_threshold: float = 0.04,
    target_speed: float = 1.5,
    vel_err_scale: float = 2.0,
    standing_point_offset: float = 0.1
) -> torch.Tensor:
    robot: RigidObject = env.scene["robot"]
    
    robot_pos_w = robot.data.root_pos_w         # 世界坐标
    target_point_world = env.hit_position_w     # 世界坐标 (num_envs, 3)
    
    standing_offset_world = torch.zeros_like(target_point_world)
    standing_offset_world[..., 0] = -standing_point_offset  # x 方向偏移
    
    standing_point_world = target_point_world + standing_offset_world
    
    target_points_pos = standing_point_world[..., 0:2]  # 只取 xy 坐标
    root_pos = robot_pos_w.unsqueeze(1)

    target_pos_diff = target_points_pos.unsqueeze(1) - root_pos[..., 0:2]
    target_pos_err = torch.sum(target_pos_diff ** 2, dim=-1)        
    near_mask = target_pos_err <= near_threshold

    # ---- velocity reward ----
    root_vel = robot.data.root_lin_vel_w.unsqueeze(1)   # 世界坐标系中的速度
    root_vel = root_vel[:, :, :2]  # only use the xy components
    
    target_dir = torch.nn.functional.normalize(target_pos_diff, dim=-1, eps=1e-6)
    target_dir_speed = torch.sum(target_dir * root_vel, dim=-1)
    
    target_vel_err = target_speed - target_dir_speed
    vel_reward = torch.exp(-vel_err_scale * target_vel_err ** 2)
    vel_reward[target_dir_speed <= 0] = 0

    # ---- mask when close ----
    vel_reward[near_mask] = 1.0
    
    return vel_reward.view(-1)

def calc_heading_quat(quat: torch.Tensor) -> torch.Tensor:
    """
    从完整的四元数中提取出 heading（朝向）分量，只保留绕 z 轴的旋转。
    
    Args:
        quat: 四元数张量，形状为 (..., 4)，格式为 [x, y, z, w]
        
    Returns:
        heading_quat: 只包含 yaw 旋转的四元数，形状与输入相同
    """
    # 输入四元数的形状
    input_shape = quat.shape
    device = quat.device
    
    # 提取四元数分量 [x, y, z, w]
    qx, qy, qz, qw = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # 方法1：通过欧拉角提取 yaw
    # 计算 yaw 角（绕 z 轴旋转）
    # yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    
    # 将 yaw 角转换回四元数（只有绕 z 轴的旋转）
    # 对于绕 z 轴的旋转：q = [0, 0, sin(yaw/2), cos(yaw/2)]
    half_yaw = yaw * 0.5
    zeros = torch.zeros_like(half_yaw)
    
    heading_quat = torch.stack([
        zeros,                    # x = 0
        zeros,                    # y = 0  
        torch.sin(half_yaw),      # z = sin(yaw/2)
        torch.cos(half_yaw)       # w = cos(yaw/2)
    ], dim=-1)
    
    return heading_quat

def walk_facing_reward_w(
    env: ManagerBasedRLEnv,
    near_threshold: float = 0.04,
    standing_point_offset: float = 0.2
) -> torch.Tensor:
    robot: RigidObject = env.scene["robot"]
    
    robot_quat_w = robot.data.root_quat_w       # 世界坐标系中的四元数
    robot_pos_w = robot.data.root_pos_w         # 世界坐标
    target_point_world = env.hit_position_w     # 世界坐标 (num_envs, 3)
    
    standing_offset_world = torch.zeros_like(target_point_world)
    standing_offset_world[..., 0] = -standing_point_offset  # x 方向偏移
    
    standing_point_world = target_point_world + standing_offset_world
    
    target_points_pos = standing_point_world[..., 0:2]  # 只取 xy 坐标
    root_pos = robot_pos_w.unsqueeze(1)

    target_pos_diff = target_points_pos.unsqueeze(1) - root_pos[..., 0:2]
    target_pos_err = torch.sum(target_pos_diff ** 2, dim=-1)        
    near_mask = target_pos_err <= near_threshold

    # ---- facing reward ----
    root_rot = robot_quat_w.unsqueeze(1)
    heading_rot = calc_heading_quat(root_rot)
    facing_dir = torch.zeros((env.num_envs, 1, 3), device=env.device)
    facing_dir[..., 0] = 1.0
    facing_dir = math_utils.quat_rotate(heading_rot, facing_dir)
    
    target_dir = torch.nn.functional.normalize(target_pos_diff, dim=-1, eps=1e-6)
    facing_err = torch.sum(target_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)

    # ---- mask when close ----
    facing_reward[near_mask] = 1.0
    
    return facing_reward.view(-1)

def homiecmd_limit_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize velocity commands that exceed their defined operational range.
    
    Ranges:
    - vx, vy, vyaw: [-1.0, 1.0]
    - base_height: [0.5, 0.8]
    """
    max_linvel = 2.0  # max linear velocity
    max_angvel = 0.5  # max angular velocity
    min_height = 0.6  # min base height
    max_height = 0.8
    cmd = env.action_manager.action[:, :4]  # 获取指令动作的相关部分，shape: (num_envs, 4)
    
    # 分别处理速度和高度指令
    vel_cmds = cmd[:, 0:2]  # vx, vy, vyaw
    angvel_cmds = cmd[:, 2]      # vyaw
    height_cmd = cmd[:, 3]      # base_height
    
    vel_excess = torch.clamp(torch.abs(vel_cmds) - max_linvel, min=0.0)
    angvel_excess = torch.clamp(torch.abs(angvel_cmds) - max_angvel, min=0.0)

    height_excess_upper = torch.clamp(height_cmd - max_height, min=0.0)
    # 计算低于下限0.5的部分
    height_excess_lower = torch.clamp(min_height - height_cmd, min=0.0)
    
    # -- 合并惩罚 --
    # 对所有超出部分取平方，并求和
    vel_penalty = torch.sum(torch.square(vel_excess), dim=1)
    angvel_penalty = torch.square(angvel_excess)
    height_penalty = torch.square(height_excess_upper) + torch.square(height_excess_lower)
    
    # 返回每个环境的总超限惩罚
    return vel_penalty+ angvel_penalty+ height_penalty

def waist_cmd_limit_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize waist commands that exceed their defined operational range."""
    pitch_limit = 0.52*4
    roll_limit = 0.52
    yaw_limit = 1.0

    waist_cmd = env.action_manager.action[:, 4:7]  # 获取指令动作的相关部分，shape: (num_envs, 4)
    waist_pitch = waist_cmd[:, 0]  
    waist_roll = waist_cmd[:, 1]     
    waist_yaw = waist_cmd[:, 2]     
    
    pitch_excess = torch.clamp(torch.abs(waist_pitch) - pitch_limit, min=0.0)
    roll_excess = torch.clamp(torch.abs(waist_roll) - roll_limit, min=0.0)
    yaw_excess = torch.clamp(torch.abs(waist_yaw) - yaw_limit, min=0.0)
    
    # -- 合并惩罚 --
    # 对所有超出部分取平方，并求和
    penalty = torch.square(pitch_excess) + torch.square(roll_excess) + torch.square(yaw_excess)
    
    # 返回每个环境的总超限惩罚
    return penalty

def stand_pose_reward(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """
    奖励机器人保持直立姿态（z轴与世界z轴对齐）
    
    Args:
        env: 环境
        std: 标准差，控制奖励衰减速度
    
    Returns:
        奖励值，形状为 [num_envs]
    """
    robot: RigidObject = env.scene["robot"]
    
    # 获取机器人base的四元数（世界坐标系）
    base_quat = robot.data.root_quat_w  # [num_envs, 4]
    
    # 定义机器人局部z轴（向上）
    local_z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, -1)
    
    # 将局部z轴旋转到世界坐标系
    robot_z_axis_world = math_utils.quat_apply(base_quat, local_z_axis)  # [num_envs, 3]
    
    # 与世界z轴 [0, 0, 1] 做点积
    # 由于世界z轴是 [0, 0, 1]，点积就是取第三个分量
    dot_product = robot_z_axis_world[:, 2]  # [num_envs]
    
    # 计算与理想值1的偏差平方
    deviation_sq = (1.0 - dot_product) ** 2
    
    # 使用高斯型奖励函数（类似参考代码的形式）
    # 当点积=1时（完全直立），reward=1.0
    # 当点积偏离1时，reward逐渐降低
    reward = 1.0 / (1.0 + deviation_sq / (std ** 2))
    
    return reward

def stand_pose_penalty_exp(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """
    惩罚机器人偏离直立姿态（z轴与世界z轴对齐）
    使用指数惩罚，倾倒时惩罚会急剧增加
    
    Args:
        env: 环境
        std: 标准差，控制惩罚增长速度（建议 0.1-0.3）
    
    Returns:
        惩罚值，形状为 [num_envs]，范围约 [-20, 0]
    """
    robot: RigidObject = env.scene["robot"]
    
    # 获取机器人base的四元数（世界坐标系）
    base_quat = robot.data.root_quat_w  # [num_envs, 4]
    
    # 定义机器人局部z轴（向上）
    local_z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, -1)
    
    # 将局部z轴旋转到世界坐标系
    robot_z_axis_world = math_utils.quat_apply(base_quat, local_z_axis)  # [num_envs, 3]
    
    # 与世界z轴 [0, 0, 1] 做点积
    dot_product = robot_z_axis_world[:, 2]  # [num_envs]，范围 [-1, 1]
    
    # 计算与理想值1的偏差
    deviation = 1.0 - dot_product  # 范围 [0, 2]，0表示完全直立，2表示完全倒立
    
    # 使用指数惩罚：倾倒时惩罚急剧增加
    # 当 deviation=0（直立）时，penalty=0
    # 当 deviation 增大时，penalty 呈指数增长
    penalty = torch.exp(deviation / std) - 1.0
    
    return penalty