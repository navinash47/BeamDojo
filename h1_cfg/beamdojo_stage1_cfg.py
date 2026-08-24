"""BeamDojo Stage 1 H1: flat physics + imagined beam heightfield (paper §IV-C.1)."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from h1_cfg.beamdojo_common import apply_play, apply_stage1
from h1_cfg.robot_spec import H1

import gymnasium as gym


@configclass
class BeamDojoStage1EnvCfg(LocomotionVelocityRoughEnvCfg):
    """Walk on a plane; scan/reward against an imagined 20 cm beam. Timeout-only."""

    def __post_init__(self):
        super().__post_init__()
        apply_stage1(self, H1)


@configclass
class BeamDojoStage1EnvCfg_PLAY(BeamDojoStage1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_play(self)


gym.register(
    id="Isaac-BeamDojo-Stage1-H1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BeamDojoStage1EnvCfg,
        "rsl_rl_cfg_entry_point": "beamdojo_agents.rsl_rl_ppo_cfg:BeamDojoPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BeamDojo-Stage1-H1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BeamDojoStage1EnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": "beamdojo_agents.rsl_rl_ppo_cfg:BeamDojoPPORunnerCfg",
    },
)
