# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, FlatPatchSamplingCfg
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors.ray_caster import RayCasterCameraCfg, patterns
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# import omni.isaac.leggedloco.leggedloco.mdp as mdp
import isaaclab.envs.mdp as mdp
from . import mdp as mymdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ActionsCfg, RewardsCfg, EventCfg, LocomotionVelocityRoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1Rewards
# from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg as OriH1RoughEnvCfg

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets import G1_MINIMAL_CFG, G1_CFG  # isort: skip

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)

# from rsl_rl_homie.rsl_rl import 
# from omni.isaac.leggedloco.utils import ASSETS_DIR

script_dir = os.path.dirname(os.path.abspath(__file__))

"""Configuration for the Unitree G1 Humanoid robot without arms."""
G1_NO_ARMS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path = os.path.join(script_dir, "assets", "robotV3", "g1_racket_29.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            # pos for badminton
            # "right_shoulder_pitch_joint": -1.3,
            # "right_shoulder_roll_joint": -0.28,
            # "right_shoulder_yaw_joint": -0.24,
            # "right_elbow_joint": 0.20,
            # pos for tennis
            "right_shoulder_pitch_joint": 0.55,
            "right_shoulder_roll_joint": -0.10,
            "right_shoulder_yaw_joint": -0.90,
            "right_elbow_joint": 0.84,
            "right_wrist_roll_joint": 0.57,
            "right_wrist_pitch_joint": 0.0, 
            "right_wrist_yaw_joint": 0.66,

        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_.*",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 100.0,  # Adjusted
                ".*_hip_roll_joint": 100.0,  # Adjusted
                ".*_hip_pitch_joint": 100.0, # Adjusted
                ".*_knee_joint": 150.0,      # Adjusted
                "waist_.*": 300.0,           # Adjusted
            },
            damping={
                ".*_hip_yaw_joint": 2.0,   # Adjusted
                ".*_hip_roll_joint": 2.0,  # Adjusted
                ".*_hip_pitch_joint": 2.0, # Adjusted
                ".*_knee_joint": 4.0,      # Adjusted
                "waist_.*": 5.0,           # Adjusted
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=40.0,  # Adjusted
            damping=2.0,     # Adjusted
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={ # Using dictionary for specific joint stiffness
                ".*_shoulder_.*": 200.0, # Adjusted
                ".*_elbow_joint": 100.0, # Adjusted
                ".*_wrist_.*": 20.0,     # Adjusted
                # Note: "hand" joints are not explicitly defined in the joint_names_expr for "arms"
                # If "hand" refers to wrist joints, then 20.0 is applied.
                # If there are separate "hand" joints not included in "*_wrist_.*",
                # you'll need to add them to joint_names_expr and define their stiffness.
            },
            damping={ # Using dictionary for specific joint damping
                ".*_shoulder_.*": 4.0,   # Adjusted
                ".*_elbow_joint": 1.0,   # Adjusted
                ".*_wrist_.*": 0.5,      # Adjusted
                # Same note as above for "hand" damping
            },
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
            },
        ),
    },
)


"""Configuration for custom terrains."""
ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=1,

    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={ # plane terrain
        "init_pos": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=1.0, 
            num_obstacles=0,
            obstacle_height_mode="fixed",
            obstacle_height_range=(1.5, 1.5), obstacle_width_range=(0.8, 1.5), 
            platform_width=2.0
        ),

    },
)

"""Rough terrains configuration."""

##
# Scene definition
##
@configclass
class TrainSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""
    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5, # TRY 9 AS WELL
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot = G1_NO_ARMS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, debug_vis=False)
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=1000.0,
        )
    )
    # object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.SphereCfg(
                    radius=0.033,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        restitution=0.60,      # 网球恢复系数 0.55~0.65
                        restitution_combine_mode="max",
                        static_friction=0.6,
                        dynamic_friction=0.5,
                    ),     
                ),
   
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=8,disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.057),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0, 0.0, 1.0)),
    )


history_length = 6

##
# Observations
##
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel,scale=0.5)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        joint_pos = ObsTerm(func=mymdp.joint_pos_rel_up)
        joint_vel = ObsTerm(func=mymdp.joint_vel_rel_up,scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        
        hit_pos_target = ObsTerm(func=mymdp.hit_pos_rel_base_target)
        # hit_pos_relative = ObsTerm(func=mymdp.hit_pos_rel_base_relative)
        time_before_hit = ObsTerm(func=mymdp.time_before_hit)
        racket_pos = ObsTerm(func=mymdp.racket_pos_rel_base)
        racket_vel = ObsTerm(func=mymdp.racket_vel_rel_base)
        racket_facing = ObsTerm(func=mymdp.racket_facing_rel_base)
        # history_length = 3

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # @configclass
    # class ProprioCfg(ObsGroup):
    #     """Observations for proprioceptive group."""
    #     base_ang_vel = ObsTerm(func=mdp.base_ang_vel,scale=0.5)
    #     projected_gravity = ObsTerm(func=mdp.projected_gravity)

    #     joint_pos = ObsTerm(func=mdp.joint_pos_rel_up)
    #     joint_vel = ObsTerm(func=mdp.joint_vel_rel_up,scale=0.05)
    #     actions = ObsTerm(func=mdp.last_action)
        
    #     hit_pos = ObsTerm(func=mdp.hit_pos_rel_base)
    #     time_before_hit = ObsTerm(func=mdp.time_before_hit)
    #     racket_pos = ObsTerm(func=mdp.racket_pos_rel_base)
    #     racket_vel = ObsTerm(func=mdp.racket_vel_rel_base)
    #     racket_facing = ObsTerm(func=mdp.racket_facing_rel_base)

    #     def __post_init__(self):
    #         self.concatenate_terms = True

    @configclass
    class CriticObsCfg(ObsGroup):
        """Observations for policy group."""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel,scale=0.5)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        joint_pos = ObsTerm(func=mymdp.joint_pos_rel_up)
        joint_vel = ObsTerm(func=mymdp.joint_vel_rel_up,scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        
        hit_pos_target = ObsTerm(func=mymdp.hit_pos_rel_base_target)
        hit_pos_relative = ObsTerm(func=mymdp.hit_pos_rel_base_relative)
        time_before_hit = ObsTerm(func=mymdp.time_before_hit)
        racket_pos = ObsTerm(func=mymdp.racket_pos_rel_base)
        # racket_vel = ObsTerm(func=mymdp.racket_vel_rel_base)
        # racket_facing = ObsTerm(func=mymdp.racket_facing_rel_base)
        # history_length = 3

        def __post_init__(self):
            # self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class HomieCfg(ObsGroup):
        """Observations for the Homie group."""

        # observation terms
        # velocity_commands = ObsTerm(func=mdp.homie_command)
        velocity_commands = ObsTerm(func=mymdp.zero_homie_command)

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2),scale=0.5)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(func=mymdp.joint_pos_rel_homie, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mymdp.joint_vel_rel_homie, noise=Unoise(n_min=-0.01, n_max=0.1),scale=0.05)
        actions = ObsTerm(func=mymdp.last_action_homie)
        # actions = ObsTerm(func=mdp.last_action)
        history_length = 6
        
        def __post_init__(self):
            self.concatenate_terms = True
            self.flatten_history_dim = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticObsCfg = CriticObsCfg()
    homie: HomieCfg = HomieCfg()
    # proprio: ProprioCfg = ProprioCfg()

## 
# Actions
##
HOMIE_ACTION_NAME=['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 
                   'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
                   'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
                   'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25)
    homie_vel= mymdp.HomieJointPositionActionCfg(asset_name="robot", joint_names=HOMIE_ACTION_NAME,scale=0.25)
    waist_pitch=mdp.JointPositionActionCfg(asset_name="robot",joint_names=["waist_pitch_joint"], scale=0.0)
    # waist_roll_yaw=mdp.JointPositionActionCfg(asset_name="robot",joint_names=["waist_roll_joint","waist_yaw_joint"], scale=0.25)
    waist_roll=mdp.JointPositionActionCfg(asset_name="robot",joint_names=["waist_roll_joint"], scale=0.0)
    waist_yaw=mdp.JointPositionActionCfg(asset_name="robot",joint_names=["waist_yaw_joint"], scale=0.25)
    left_arm_ik=mdp.DifferentialInverseKinematicsActionCfg(
        scale=0.01,
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.1, 0.0, 0.0),  
            rot=(1.0, 0.0, 0.0, 0.0), 
        ),
        asset_name="robot",  
        joint_names=["left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_.*"],
        # body_name="racket_link",  
        body_name="left_wrist_yaw_link",
        controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    )
    right_arm_ik=mdp.DifferentialInverseKinematicsActionCfg(
        scale=0.01,  
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
            # pos=(0.14, 0.0, 0.15),  
            pos=(0.0, 0.0, 0.0),  
            rot=(1.0, 0.0, 0.0, 0.0), 
        ), 
        asset_name="robot",  
        joint_names=["right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_.*"],        
        body_name="right_wrist_yaw_link",
        controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    )
##
# Events (for domain randomization)
##
@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (-0.0,0.0),
                "y": (-0.0,0.0),
                "z": (-0.0,0.0),
                "roll": (-0.0,0.0),
                "pitch": (-0.0,0.0),
                "yaw": (-0.0,0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_hit = EventTerm(
        func=mymdp.reset_hit, 
        mode="reset",
        params={
            "pos_range_x":(0.15, 0.20),
            "pos_range_y":(-0.75, -0.70),
            "pos_range_z":(-0, -0.01)
            # "pos_range_z":(0.1, 0.2)
        }
    )

    reset_ball = EventTerm(
        func=mymdp.reset_ball,  # 你的自定义函数 
        mode="reset",                 # 在reset时触发
        params={
            "asset_cfg": SceneEntityCfg("object"),
        }
    )



@configclass
class G1NewRewardCfg:
    """Reward terms for the badminton-playing robot task."""
    
    track_swing_position = RewTerm(
        func=mymdp.track_swing_position_at_impact, weight=6400.0,
        params={"std": 0.05} # 位置误差的指数核标准差 (m)
    )

    track_swing_velocity = RewTerm(
        func=mymdp.track_swing_velocity_at_impact, weight=5000.0,
        params={"std": 1.0, "time_window": 0.02} # 速度误差的指数核标准差 (m/s)
    )
    track_swing_velocity_wide = RewTerm(
        func=mymdp.track_swing_velocity_at_impact, weight=50.0 * 8,
        params={"std": 1.2, "time_window": 0.12}
    )
    # track_swing_vel_dir = RewTerm(
    #     func=mdp.track_swing_velocity_direction_at_impact,
    #     weight=3000.0,
    #     params={"std": 0.1}
    # )

    # track_swing_vel_mag = RewTerm(
    #     func=mdp.track_swing_velocity_magnitude_at_impact,
    #     weight=2000.0,
    #     params={"std": 0.5, "target_speed": 4.0}
    # )

    track_swing_orientation = RewTerm(
        func=mymdp.track_swing_orientation_at_impact, weight=1000.0,
        params={"std": 0.3} # 姿态误差的指数核标准差 (rad)
    )

    # maintain_default_pos_after = RewTerm(
    #     func=mdp.maintain_default_pose_after_impact, weight=-6.0)
    maintain_default_pos_before = RewTerm(
        func=mymdp.maintain_default_pose_before_impact, weight=-30.0)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces",body_names="^(?!.*ankle).*"), "threshold": 1.0},
    )    
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1e-4)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_power = RewTerm(
        func=mymdp.power_penalty,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            ".*_elbow_joint", ".*_wrist_.*", ".*_shoulder_.*", "waist_.*"])},
    )
    # stand_pose_hold = RewTerm(
    #     func=mdp.stand_pose_reward,
    #     weight = 14.0,
    #     params={"std": 0.2}
    # )
    # stand_pose_hold = RewTerm(
    #     func=mdp.stand_pose_penalty_exp,
    #     weight = -14.0,
    #     params={"std": 0.2}
    # )
    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)
    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    homie_cmd_limits = RewTerm(
        func=mymdp.homiecmd_limit_penalty,
        weight=-1.0,
    )
    waist_cmd_limits = RewTerm(
        func=mymdp.waist_cmd_limit_penalty,
        weight=-1.0
    )
    homie_cmd_rate = RewTerm(
        func=mymdp.homiecmd_rate_l2, 
        weight=-1e-4
    )  
    # walk_pos = RewTerm(
    #     func=mdp.walk_position_reward_w,
    #     weight=100.0 / 10,
    # )
    # walk_velo = RewTerm(
    #     func=mdp.walk_velocity_reward_w,
    #     weight=200.0 / 10
    # )
    # walk_face = RewTerm(
    #     func=mdp.walk_facing_reward_w,
    #     weight=100.0 / 10
    # )
#
# Commands
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mymdp.UniformVelocityCommandVisualizeCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mymdp.UniformVelocityCommandVisualizeCfg.Ranges(
            lin_vel_x=(0., 0.), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0.0, 0.0), heading=(-0.0,0.0)
        ),
    )
    # no_velocity = mdp.VisualNullCommandCfg(asset_name="robot",debug_vis=True)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.25}
    )
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", 
    #                                         #  body_names=[".*shoulder.*",".*elbow.*",".*hip.*",".*knee.*","pelvis.*"]),"threshold": 1},    
    #                                          body_names=[".*hip.*",".*knee.*","pelvis.*"]),"threshold": 1},    
    # )
    # object_landed = DoneTerm(func=mdp.object_landed)
    
    

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    pass

@configclass
class G1TennisEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: TrainSceneCfg = TrainSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = G1NewRewardCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 3.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physics_material.friction_combine_mode = "average"
        self.sim.physics_material.restitution_combine_mode = "max"


        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1TennisEnvCfg_PLAY(G1TennisEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 10
        self.scene.env_spacing = 0.0
        self.episode_length_s = 240.0
        # self.scene.terrain.terrain_type="usd",
        # self.scene.terrain.usd_path="/home/xunyang/Desktop/obj_test/town_resized.usd",
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
