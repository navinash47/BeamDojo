"""BeamDojo Stage 2 G1: colliding beam, fall/off-terrain dones, double-critic runner."""

from __future__ import annotations

from isaaclab.utils import configclass

from h1_cfg.beamdojo_common import apply_play, apply_stage2
from h1_cfg.beamdojo_env_base import BeamDojoEnvCfg
from h1_cfg.robot_spec import G1

import gymnasium as gym


@configclass
class BeamDojoStage2G1EnvCfg(BeamDojoEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_stage2(self, G1, stones=False)


@configclass
class BeamDojoStage2G1EnvCfg_PLAY(BeamDojoStage2G1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_play(self)


gym.register(
    id="Isaac-BeamDojo-Stage2-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BeamDojoStage2G1EnvCfg,
        "rsl_rl_cfg_entry_point": "beamdojo_agents.rsl_rl_ppo_cfg:BeamDojoG1Stage2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BeamDojo-Stage2-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BeamDojoStage2G1EnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": "beamdojo_agents.rsl_rl_ppo_cfg:BeamDojoG1Stage2PPORunnerCfg",
    },
)


@configclass
class BeamDojoStage2G1StonesEnvCfg(BeamDojoEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_stage2(self, G1, stones=True)


@configclass
class BeamDojoStage2G1StonesEnvCfg_PLAY(BeamDojoStage2G1StonesEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_play(self)


gym.register(
    id="Isaac-BeamDojo-Stage2-G1-Stones-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BeamDojoStage2G1StonesEnvCfg,
        "rsl_rl_cfg_entry_point": "beamdojo_agents.rsl_rl_ppo_cfg:BeamDojoG1Stage2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BeamDojo-Stage2-G1-Stones-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BeamDojoStage2G1StonesEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": "beamdojo_agents.rsl_rl_ppo_cfg:BeamDojoG1Stage2PPORunnerCfg",
    },
)
