# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def joint_pos_rel_up(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_lab = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    lab2gym_reindex = [11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
    joint_pos_gym = joint_pos_lab[:,lab2gym_reindex]
    return joint_pos_gym

def joint_vel_rel_up(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel_lab = asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
    lab2gym_reindex = [11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
    joint_vel_gym = joint_vel_lab[:, lab2gym_reindex]
    return joint_vel_gym

# def last_action_up(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:

#     """The last input action to the environment.

#     The name of the action term for which the action is required. If None, the
#     entire action tensor is returned.
#     """
#     lab_action = env.action_manager.action
#     lab2gym_reindex = [11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
#     gym_action = lab_action[:,lab2gym_reindex]
#     return gym_action

# the 3d position of the object relative to the robot base
def hit_pos_rel_base_target(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    # calculate the relative position
    relative_pos = env.hit_position
    return relative_pos

def hit_pos_rel_base_relative(
    env: ManagerBasedEnv, 
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    击球点相对于 base 的位置（动态计算，考虑 base 的平移和旋转）
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    
    # 获取当前状态
    base_pos_w = robot.data.root_pos_w  # [num_envs, 3]
    base_quat_w = robot.data.root_quat_w  # [num_envs, 4]
    hit_pos_w = env.hit_position_w  # [num_envs, 3]
    
    # 世界坐标系下的相对位置
    relative_pos_w = hit_pos_w - base_pos_w
    
    # 转换到 base 坐标系
    base_quat_inv = math_utils.quat_inv(base_quat_w)
    relative_pos_base = math_utils.quat_apply(base_quat_inv, relative_pos_w)
    
    return relative_pos_base

def time_before_hit(env: ManagerBasedEnv) -> torch.Tensor:
    """
    计算距离预定拦截时刻的剩余时间。
    这个函数是一个观察项，它读取预先计算好的目标时刻并进行减法。
    """
    target_intercept_time = env.target_intercept_time.view(env.num_envs, 1)  # 确保是列向量

    current_times = env.episode_length_buf.view(env.num_envs, 1)  * 0.02
    # print("current_times: ", target_intercept_time)
    # 3. 计算剩余时间。
    remaining = target_intercept_time - current_times
    # time_remaining_tensor = torch.clamp(remaining, min=0.0)
    time_remaining_tensor = torch.clamp(remaining, min=-3.0, max=3.0)
    if env.debug_mode:
        print("time_remaining_tensor: ", remaining)
    return time_remaining_tensor.view(env.num_envs, 1)  # 确保输出是一个列向量
    # return remaining.view(env.num_envs, 1)  # 确保输出是一个列向量


def racket_pos_rel_base(env: ManagerBasedEnv) -> torch.Tensor:
    """Position of the racket in the robot base frame."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene["robot"]

    racket_pos = robot.data.body_link_pos_w[:, 37]
    base_pos = robot.data.root_pos_w
    base_quat = robot.data.root_quat_w

    racket_pos_b = math_utils.quat_apply_inverse(base_quat, racket_pos - base_pos)
    return racket_pos_b

def racket_vel_rel_base(env: ManagerBasedEnv) -> torch.Tensor:
    """Velocity of the racket in the robot base frame."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene["robot"]

    racket_vel_w = robot.data.body_link_vel_w[:, 37,:3]
    base_quat = robot.data.root_quat_w

    racket_vel_b = math_utils.quat_apply_inverse(base_quat, racket_vel_w)
    return racket_vel_b

def racket_facing_rel_base(env: ManagerBasedEnv) -> torch.Tensor:
    """Facing direction of the racket in the robot base frame."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene["robot"]
    racket_quat_w = robot.data.body_link_quat_w[:, 37]  # [num_envs, 4]
    racket_local_normal = torch.tensor([0.0, -1.0, 0.0], device=env.device).expand(env.num_envs, -1) # <-- CORRECTED LINE
    robot_base_quat_w = robot.data.root_quat_w  # [num_envs, 4]
    racket_facing_b = math_utils.quat_apply_inverse(
        robot_base_quat_w, 
        math_utils.quat_apply(racket_quat_w, racket_local_normal)
    )
    return racket_facing_b

def homie_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """An empty command tensor with the same shape as the action space."""
    # homie_command_tensor = torch.zeros((env.num_envs, 4), device=env.device, dtype=env.action_manager.action.dtype)
    # homie_command_tensor[:,3] = 0.78  # set the height to 0.75
    # homie_command_tensor = env.action_manager.action[:, :4]
    homie_command_tensor = env.action_manager.get_term("homie_vel").raw_actions
    return homie_command_tensor

def zero_homie_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """An empty command tensor with the same shape as the action space."""
    homie_command_tensor = torch.zeros((env.num_envs, 4), device=env.device, dtype=env.action_manager.action.dtype)
    homie_command_tensor[:,3] = 0.78  # set the height to 0.75
    # homie_command_tensor = env.action_manager.action[:, :4]
    # homie_command_tensor = env.action_manager.get_term("homie_vel").raw_actions
    return homie_command_tensor


def joint_pos_rel_homie(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_lab = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    lab2gym_reindex = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
    joint_pos_gym = joint_pos_lab[:,lab2gym_reindex]
    return joint_pos_gym

def joint_vel_rel_homie(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel_lab = asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
    lab2gym_reindex = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
    joint_vel_gym = joint_vel_lab[:, lab2gym_reindex]
    return joint_vel_gym

# def last_action_homie(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
#     """The last input action to the environment.

#     The name of the action term for which the action is required. If None, the
#     entire action tensor is returned.
#     """
#     lab_action = env.action_manager.action
#     lab2gym_reindex = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
#     gym_action = lab_action[:,lab2gym_reindex]
#     homie_action = gym_action[:, :12]  # only the first 12 actions are used in homie
#     return homie_action

def last_action_homie(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    # homie_action = env.action_manager._terms["homie_vel"].homie_action
    homie_action = env.action_manager.get_term("homie_vel").homie_action
    return homie_action