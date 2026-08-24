"""BeamDojo Stage 1 G1: same dual-terrain as H1, Unitree G1 + 12 lower-body actions."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from h1_cfg.beamdojo_common import apply_play, apply_stage1
from h1_cfg.robot_spec import G1

import gymnasium as gym


@configclass
class BeamDojoStage1G1EnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_stage1(self, G1)


@configclass
class BeamDojoStage1G1EnvCfg_PLAY(BeamDojoStage1G1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_play(self)


gym.register(
    id="Isaac-BeamDojo-Stage1-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BeamDojoStage1G1EnvCfg,
        "rsl_rl_cfg_entry_point": "beamdojo_agents.rsl_rl_ppo_cfg:BeamDojoG1PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BeamDojo-Stage1-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BeamDojoStage1G1EnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": "beamdojo_agents.rsl_rl_ppo_cfg:BeamDojoG1PPORunnerCfg",
    },
)
