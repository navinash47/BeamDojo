# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import H1RoughEnvCfg

_BEAM_USD_PATH = "/home/lily-hcrlab/issaclab/IsaacLab/scripts/beamdojo/props/beam_balance.usd"


@configclass
class H1BeamEnvCfg(H1RoughEnvCfg):
    """Configuration for locomotion on a narrow balance beam."""

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # scene layout tuned for single shared beam asset
        self.scene.num_envs = 512
        self.scene.env_spacing = 5.5
        self.scene.replicate_physics = True

        # use a flat plane terrain as a safety fall-back
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/beam_ground",
            terrain_type="plane",
            env_spacing=self.scene.env_spacing,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )
        self.sim.physics_material = self.scene.terrain.physics_material

        # add a narrow balance beam per environment
        self.scene.beam = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Beam",
            spawn=UsdFileCfg(usd_path=_BEAM_USD_PATH),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )

        # no height scan is needed for the beam setup
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # disable terrain curriculum and aggressive disturbances – focus on balance
        self.curriculum.terrain_levels = None
        self.events.push_robot = None
        self.events.base_external_force_torque = None

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

        # restrict commanded motion to forward progression with minimal lateral drift
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        if hasattr(self.commands.base_velocity.ranges, "heading"):
            self.commands.base_velocity.ranges.heading = None

        # encourage upright posture and smooth control while keeping reward structure lean
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.dof_torques_l2.weight = -2.5e-6
        self.rewards.action_rate_l2.weight = -0.01

        # terminate on loss of balance—low torso height or excessive body tilt 
        self.terminations.base_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.35, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.base_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(35.0), "asset_cfg": SceneEntityCfg("robot")},
        )

        # terminate quickly when torso leaves beam height band
        base_contact_cfg = self.terminations.base_contact.params.get("sensor_cfg")
        if base_contact_cfg is not None:
            base_contact_cfg.body_names = ".*torso_link"


@configclass
class H1BeamEnvCfg_PLAY(H1BeamEnvCfg):
    """Lighter configuration for running the beam task interactively."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # smaller scene for interactive runs
        self.scene.num_envs = 64
        self.scene.env_spacing = 6.0
        self.episode_length_s = 40.0

        # disable corruption and additional randomization for play
        self.observations.policy.enable_corruption = False
        self.events.reset_base.params["pose_range"]["y"] = (-0.05, 0.05)
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.2, 0.2)
