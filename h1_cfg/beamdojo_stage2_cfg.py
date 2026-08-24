"""BeamDojo Stage 2 H1: colliding beam/stones, fall + off-terrain terminate, curriculum."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from h1_cfg.beamdojo_common import apply_play, apply_stage2
from h1_cfg.robot_spec import H1

import gymnasium as gym


@configclass
class BeamDojoStage2EnvCfg(LocomotionVelocityRoughEnvCfg):
    """Hard beam: collision cuboid, catcher plane, off-terrain / fall dones."""

    def __post_init__(self):
        super().__post_init__()
        apply_stage2(self, H1, stones=False)


@configclass
class BeamDojoStage2EnvCfg_PLAY(BeamDojoStage2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_play(self)


@configclass
class BeamDojoStage2StonesEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_stage2(self, H1, stones=True)


@configclass
class BeamDojoStage2StonesEnvCfg_PLAY(BeamDojoStage2StonesEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_play(self)


gym.register(
    id="Isaac-BeamDojo-Stage2-H1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BeamDojoStage2EnvCfg,
        "rsl_rl_cfg_entry_point": "beamdojo_agents.rsl_rl_ppo_cfg:BeamDojoStage2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BeamDojo-Stage2-H1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BeamDojoStage2EnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": "beamdojo_agents.rsl_rl_ppo_cfg:BeamDojoStage2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BeamDojo-Stage2-H1-Stones-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BeamDojoStage2StonesEnvCfg,
        "rsl_rl_cfg_entry_point": "beamdojo_agents.rsl_rl_ppo_cfg:BeamDojoStage2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BeamDojo-Stage2-H1-Stones-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BeamDojoStage2StonesEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": "beamdojo_agents.rsl_rl_ppo_cfg:BeamDojoStage2PPORunnerCfg",
    },
)
