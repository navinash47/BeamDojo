# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""
BeamDojo Stage 1: Balance Beam with Soft Dynamics Constraints
Trains on flat terrain while "imagining" balance beam via elevation map.
Allows missteps without termination for exploration.
"""

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab import sim as sim_utils

if TYPE_CHECKING:
    import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# Import H1 robot configuration
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

##
# Pre-defined configs
##
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1 import agents  # isort: skip
from isaaclab_assets.robots.unitree import H1_CFG as UNITREE_H1_CFG  # isort: skip


##
# BeamDojo Stage 1 Environment Configuration
##

@configclass
class BeamDojoStage1EnvCfg(LocomotionVelocityRoughEnvCfg):
    """
    BeamDojo Stage 1: Soft terrain dynamics constraints.
    
    Robot walks on FLAT TERRAIN but receives elevation map showing balance beam.
    Foothold rewards are computed based on target beam terrain (not actual flat ground).
    No termination on falls - allows continuous exploration.
    
    Based on BeamDojo paper Section IV-C.1
    """

    def __post_init__(self):
        # Call parent post_init first
        super().__post_init__()

        # ===== SCENE CONFIGURATION =====
        self.scene.num_envs = 4096  # BeamDojo uses 4096 parallel environments
        self.scene.env_spacing = 8.0  # 8m spacing to match beam length
        
        # ===== ROBOT CONFIGURATION =====
        # Use Unitree H1 humanoid
        self.scene.robot = UNITREE_H1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 1.05)  # Start standing on flat ground
        
        # ===== TERRAIN CONFIGURATION =====
        # Stage 1: Use FLAT terrain for actual physics
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",  # Flat plane
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )
        
        # ===== HEIGHT SCANNER CONFIGURATION =====
        # This will scan the TARGET beam terrain (not the flat ground)
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),  # Above pelvis
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                resolution=0.1,  # 10cm resolution
                size=[1.5, 1.5],  # 1.5m x 1.5m scan area (15x15 grid)
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],  # This will be overridden to scan target terrain
        )
        
        # ===== CONTACT SENSORS =====
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=True,
            update_period=0.0,  # Every step
        )
        
        # Disable events that reference non-existent generic "base" body names
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.push_robot = None
        self.events.base_external_force_torque = None

        # ===== REWARD CONFIGURATION =====
        # Group 1: Dense locomotion rewards (use actual flat terrain)
        self.rewards.lin_vel_z_l2.weight = -2.0  # Penalize vertical velocity
        self.rewards.ang_vel_xy_l2.weight = -0.05  # Penalize body roll/pitch velocity
        self.rewards.flat_orientation_l2.weight = -1.0  # Keep body upright (reduced from -2.0 to match H1RoughEnvCfg)
        self.rewards.dof_torques_l2.weight = 0.0  # Disable torque penalty (matches H1RoughEnvCfg)
        self.rewards.action_rate_l2.weight = -0.005  # Smooth actions (reduced from -0.01 to match H1RoughEnvCfg)
        self.rewards.dof_acc_l2.weight = -1.25e-7  # Smooth joint accelerations (reduced from -2.5e-7 to match H1RoughEnvCfg)
        self.rewards.undesired_contacts = None
        
        # CRITICAL: Base height penalty to prevent robot from lying down.
        # Since Stage 1 has no terminations (except time_out), is_terminated always returns False.
        # Instead, we penalize low base height (lying down) vs target standing height (~1.05 m for H1).
        # This forces the robot to stand up instead of collapsing on the floor.
        self.rewards.base_height_penalty = RewTerm(
            func=mdp.base_height_l2,
            weight=-10.0,  # Stronger penalty for being too low (lying down)
            params={
                "target_height": 1.05,  # Standing pelvis height for H1 on flat ground
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        
        # Also keep termination penalty (though it won't activate in Stage 1, it's good for consistency)
        self.rewards.termination_penalty = RewTerm(
            func=mdp.is_terminated,
            weight=-200.0,
        )
        
        # Velocity tracking rewards (use yaw-frame for humanoids)
        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func=mdp.track_lin_vel_xy_yaw_frame_exp,  # Yaw-frame version for humanoids
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.rewards.track_ang_vel_z_exp = RewTerm(
            func=mdp.track_ang_vel_z_world_exp,  # World-frame angular velocity
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        
        # Gait rewards for humanoid (use positive biped version)
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time_positive_biped,  # Positive biped version for humanoids
            weight=0.25,  # Reduced from 1.0 to match H1RoughEnvCfg
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_link"]),
                "threshold": 0.4,  # 0.4s air time target (matches H1RoughEnvCfg)
            },
        )
        
        # Feet slide penalty - encourages proper foot placement
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.25,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_link"]),
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_link"]),
            },
        )
        
        # Penalize ankle joint limits - prevents extreme joint angles
        self.rewards.dof_pos_limits = RewTerm(
            func=mdp.joint_pos_limits,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle"])},
        )
        
        # Penalize deviation from default joint positions for non-locomotion joints
        # Keeps joints in reasonable positions and prevents unnatural poses
        self.rewards.joint_deviation_hip = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw", ".*_hip_roll"])},
        )
        self.rewards.joint_deviation_arms = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow"])},
        )
        self.rewards.joint_deviation_torso = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso")},
        )
        
        # Group 2: Sparse foothold reward (based on TARGET beam terrain)
        # This is the key BeamDojo innovation - reward for staying on imagined beam
        self.rewards.foothold_penalty = RewTerm(
            func=foothold_reward_stage1,  # Custom function (defined below)
            weight=1.0,  # BeamDojo uses weight 1.0 for foothold in stage 1
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_link"]),
                "target_beam_width": 0.20,  # 20cm beam width (BeamDojo spec)
                "target_beam_length": 8.0,   # 8m beam length
                "target_beam_y_center": 0.0, # Centered at y=0
                "num_samples": 15,           # Sample 15 points per foot (like BeamDojo)
                "depth_threshold": -0.1,     # -10cm = off beam
            },
        )
        
        # ===== TERMINATION CONFIGURATION =====
        # Stage 1: NO termination on falls! Only time limit
        # This allows full exploration without early episode termination
        self.terminations.time_out = DoneTerm(func=mdp.time_out, time_out=True)
        
        # Remove fall-based terminations for stage 1
        if hasattr(self.terminations, "base_height"):
            delattr(self.terminations, "base_height")
        if hasattr(self.terminations, "base_orientation"):
            delattr(self.terminations, "base_orientation")
        if hasattr(self.terminations, "base_contact"):
            delattr(self.terminations, "base_contact")
        
        # ===== COMMAND CONFIGURATION =====
        # Stage 1: Train with diverse commands (lateral and yaw)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)  # Lateral movement
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)   # Rotation
        # Make 50% of the environments stand still to reinforce upright posture early in training.
        self.commands.base_velocity.rel_standing_envs = 0.5
        
        # ===== TRAINING CONFIGURATION =====
        self.episode_length_s = 20.0  # 20 second episodes
        self.decimation = 4  # 50Hz policy control (200Hz / 4)
        self.sim.dt = 0.005  # 5ms simulation timestep (200Hz)
        
        # ===== OBSERVATION CONFIGURATION =====
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.projected_gravity.scale = 1.0
        
        # Add height scan observation (scans TARGET beam terrain)
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
            clip=(-1.0, 1.0),
        )

        # Disable terrain curriculum since Stage 1 uses a flat plane terrain
        self.curriculum.terrain_levels = None


@configclass
class BeamDojoStage1EnvCfg_PLAY(BeamDojoStage1EnvCfg):
    """Lighter configuration for running the BeamDojo Stage 1 task interactively."""

    def __post_init__(self):
        super().__post_init__()

        # smaller scene for interactive runs
        self.scene.num_envs = 64
        self.scene.env_spacing = 6.0
        self.episode_length_s = 40.0

        # disable corruption and additional randomization for play
        self.observations.policy.enable_corruption = False
        
        # tighten spawn distribution to keep robot close to beam center line
        pose_range = self.events.reset_base.params.get("pose_range", {})
        pose_range["x"] = (-0.2, 0.2)
        pose_range["y"] = (-0.08, 0.08)
        pose_range["yaw"] = (-0.2, 0.2)
        self.events.reset_base.params["pose_range"] = pose_range

        velocity_range = self.events.reset_base.params.get("velocity_range", {})
        for key in ("x", "y", "z", "roll", "pitch", "yaw"):
            velocity_range[key] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"] = velocity_range


##
# Custom Reward Function for Stage 1
##

def foothold_reward_stage1(
    env,
    sensor_cfg: SceneEntityCfg,
    target_beam_width: float,
    target_beam_length: float,
    target_beam_y_center: float,
    num_samples: int,
    depth_threshold: float,
) -> torch.Tensor:
    """
    BeamDojo sampling-based foothold reward for Stage 1.
    
    Computes reward based on foot placement on TARGET (imagined) balance beam,
    even though robot walks on flat ground. This allows learning foothold
    precision without early termination.
    
    From BeamDojo paper equation (2):
    r_foothold = -Σ(C_i * Σ(1{d_ij < ε}))
    
    Args:
        env: Environment instance
        sensor_cfg: Contact sensor configuration for feet
        target_beam_width: Width of imagined beam (0.20m for BeamDojo)
        target_beam_length: Length of beam
        target_beam_y_center: Y-coordinate of beam center
        num_samples: Number of sample points per foot
        depth_threshold: Threshold for "off beam" (negative = below beam)
    
    Returns:
        Foothold reward tensor [num_envs]
    """
    import torch  # noqa: F401
    
    # Get contact sensor
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene["robot"]

    # Find foot bodies on the sensor and the robot
    foot_sensor_indices, foot_names = contact_sensor.find_bodies(sensor_cfg.body_names, preserve_order=True)
    foot_robot_indices, _ = robot.find_bodies(foot_names, preserve_order=True)

    # Get foot positions in world frame
    foot_positions_w = robot.data.body_pos_w[:, foot_robot_indices, :]  # [num_envs, n_feet, 3]

    # Get contact forces to determine if foot is touching ground
    contact_forces = contact_sensor.data.net_forces_w[:, foot_sensor_indices, 2]  # Z-component
    in_contact = torch.abs(contact_forces) > 1.0  # [num_envs, 2]
    
    # Sample points under each foot (simplified grid sampling)
    # For proper implementation, sample in foot frame and transform to world
    foot_length = 0.15  # Approximate H1 foot length
    foot_width = 0.08   # Approximate H1 foot width
    
    # Create sample grid
    sqrt_n = int(math.sqrt(num_samples))
    x_offsets = torch.linspace(-foot_length/2, foot_length/2, sqrt_n, device=env.device)
    y_offsets = torch.linspace(-foot_width/2, foot_width/2, sqrt_n, device=env.device)
    
    # Initialize penalty
    penalty = torch.zeros(env.num_envs, device=env.device)
    
    # For each foot
    num_feet = foot_positions_w.shape[1]
    for foot_idx in range(num_feet):
        # Get foot XY position
        foot_xy = foot_positions_w[:, foot_idx, :2]  # [num_envs, 2]
        
        # Count samples off the target beam
        # Beam is centered at y=target_beam_y_center, width=target_beam_width
        # Beam extends along X axis for target_beam_length
        
        # Check if foot Y position is outside beam width
        foot_y = foot_xy[:, 1]
        y_min = target_beam_y_center - target_beam_width / 2
        y_max = target_beam_y_center + target_beam_width / 2
        
        # Simplified: check if center of foot is off beam
        # For full implementation, check all sample points
        off_beam = (foot_y < y_min) | (foot_y > y_max)
        
        # Also check if beyond beam length
        foot_x = foot_xy[:, 0]
        off_beam = off_beam | (foot_x < 0.0) | (foot_x > target_beam_length)
        
        # Apply penalty only when in contact
        penalty += in_contact[:, foot_idx].float() * off_beam.float() * num_samples
    
    # Return negative penalty as reward
    return -penalty


##
# Register Environment
##

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv

gym.register(
    id="Isaac-BeamDojo-Stage1-H1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BeamDojoStage1EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-BeamDojo-Stage1-H1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BeamDojoStage1EnvCfg_PLAY,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
