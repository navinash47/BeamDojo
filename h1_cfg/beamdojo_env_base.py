"""Declared BeamDojo env nested cfgs so Hydra keeps foothold / beam / stones.

Isaac Lab 2.3.2 ``class_to_dict`` dumps ``cfg.__dict__``. ``from_dict`` then
raises ``KeyError`` for keys that are not attributes of a *fresh* nested cfg
(``update_class_from_dict``). Official H1/G1 declare extras on RewardsCfg
subclasses for the same reason. Stage 1/2 H1/G1 env classes inherit
``BeamDojoEnvCfg`` instead of the ANYmal ``LocomotionVelocityRoughEnvCfg``.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    CurriculumCfg,
    EventCfg,
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg,
    RewardsCfg,
    TerminationsCfg,
)

from h1_cfg.scene_props import STONE_COUNT


@configclass
class BeamDojoSceneCfg(MySceneCfg):
    """Parent locomotion scene plus BeamDojo task geometry."""

    task_beam = None
    catcher = None
    task_stone_0 = None
    task_stone_1 = None
    task_stone_2 = None
    task_stone_3 = None
    task_stone_4 = None
    task_stone_5 = None
    task_stone_6 = None
    task_stone_7 = None
    task_stone_8 = None
    task_stone_9 = None
    task_stone_10 = None
    task_stone_11 = None
    task_stone_12 = None
    task_stone_13 = None
    task_stone_14 = None
    task_stone_15 = None
    task_stone_16 = None
    task_stone_17 = None
    task_stone_18 = None
    task_stone_19 = None
    task_stone_20 = None
    task_stone_21 = None
    task_stone_22 = None
    task_stone_23 = None


if STONE_COUNT != 24:
    raise RuntimeError(f"Declare task_stone_0..{STONE_COUNT - 1} on BeamDojoSceneCfg (have 0..23).")


@configclass
class BeamDojoRewardsCfg(RewardsCfg):
    """ANYmal locomotion rewards plus BeamDojo / H1 / G1 extras."""

    foothold_penalty = None
    feet_slide = None
    joint_deviation_hip = None
    joint_deviation_arms = None
    joint_deviation_torso = None
    joint_deviation_fingers = None
    base_height_penalty = None
    termination_penalty = None


@configclass
class BeamDojoEventCfg(EventCfg):
    init_beamdojo = None
    reset_beamdojo = None
    disable_ground = None


@configclass
class BeamDojoTerminationsCfg(TerminationsCfg):
    base_height = None
    base_orientation = None
    off_terrain = None


@configclass
class BeamDojoCurriculumCfg(CurriculumCfg):
    beam_width = None


@configclass
class BeamDojoEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Shared H1/G1 Stage 1/2 env: typed scene, rewards, events, dones, curriculum."""

    scene: BeamDojoSceneCfg = BeamDojoSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: BeamDojoRewardsCfg = BeamDojoRewardsCfg()
    events: BeamDojoEventCfg = BeamDojoEventCfg()
    terminations: BeamDojoTerminationsCfg = BeamDojoTerminationsCfg()
    curriculum: BeamDojoCurriculumCfg = BeamDojoCurriculumCfg()
