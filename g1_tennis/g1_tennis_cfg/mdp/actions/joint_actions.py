# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import os

from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg

from isaaclab.envs.mdp import JointAction

class HomieJointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.HomieJointPositionActionCfg
    """The configuration of the action term."""

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    def __init__(self, cfg: actions_cfg.HomieJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device) # homie command
        self._processed_actions = torch.zeros_like(self.raw_actions)
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        homie_policy_path = os.path.join(script_dir, "..", "..", "assets", "homie.pt")
        self.homie_policy = torch.jit.load(homie_policy_path).to(self.device)
        self.homie_action = torch.zeros(self.num_envs, 12, device=self.device) # homie command

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions # homie command
        homie_obs = self._env.obs_buf['homie'].view(self.num_envs, -1) # homie observation
        self.homie_action = self.homie_policy(homie_obs)
        # apply the affine transformations
        self._processed_actions = self.homie_action * self._scale + self._offset
