#!/usr/bin/env python3
"""BeamDojo Stage 1 training entry point for RSL-RL.

This script mirrors the generic `train.py` pipeline while pre-configuring it for the
BeamDojo Stage 1 curriculum (flat terrain with imagined beam constraints).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# -- Constants -------------------------------------------------------------------------------------------------------
STAGE1_TASK_ID = "Isaac-BeamDojo-Stage1-H1-v0"
STAGE1_DEFAULT_EXPERIMENT = "beamdojo_stage1"
STAGE1_DEFAULT_SAVE_INTERVAL = 100
STAGE1_DEFAULT_MAX_ITERS = 10_000


def _ensure_beamdojo_stage1_registered():
    """Import BeamDojo Stage 1 task ensuring `agents` module is seeded before registration."""
    module_name = "isaaclab_tasks.manager_based.locomotion.velocity.config.h1.beamdojo_stage1_cfg"
    if module_name in sys.modules:
        return

    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.loader is None:
        raise ImportError(
            "Cannot locate BeamDojo Stage 1 config module. "
            "Please verify that the file exists and is discoverable by Python."
        )

    module = importlib.util.module_from_spec(spec)
    module.__dict__["agents"] = importlib.import_module(
        "isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents"
    )
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


# -- CLI -------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="BeamDojo Stage 1 training with RSL-RL.")
parser.add_argument(
    "--stage",
    type=int,
    choices=[1],
    default=1,
    help="Training stage to execute (only Stage 1 is currently supported).",
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default=STAGE1_TASK_ID,
    help=f"Gym registry ID for the task to train (defaults to {STAGE1_TASK_ID}).",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument(
    "--max_iterations",
    type=int,
    default=STAGE1_DEFAULT_MAX_ITERS,
    help="RL policy training iterations (defaults to 10k for BeamDojo Stage 1).",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


if args_cli.stage != 1:
    raise ValueError("train_beamdojo.py currently only supports Stage 1 training.")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ensure Isaac Lab extensions are loaded so omni.isaac.lab namespace is available
import isaaclab  # noqa: F401

# ensure BeamDojo Stage 1 environment is registered (requires SimulationApp to be live)
_ensure_beamdojo_stage1_registered()


# -- Sanity check RSL-RL version -------------------------------------------------------------------------------------
import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)


# -- Imports that require the app ------------------------------------------------------------------------------------
import os
from datetime import datetime

import gymnasium as gym
import torch

import omni
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _apply_stage1_defaults(agent_cfg: RslRlBaseRunnerCfg):
    """Override runner configuration with BeamDojo Stage 1 defaults unless user overrides via CLI."""
    # experiment naming
    if not getattr(agent_cfg, "experiment_name", None) or agent_cfg.experiment_name == "h1_rough":
        agent_cfg.experiment_name = STAGE1_DEFAULT_EXPERIMENT

    if not getattr(agent_cfg, "run_name", None):
        agent_cfg.run_name = ""

    # stage-specific training cadence
    agent_cfg.save_interval = STAGE1_DEFAULT_SAVE_INTERVAL

    # algorithm hyper-parameters (align with BeamDojo Stage 1 spec)
    algorithm_cfg = getattr(agent_cfg, "algorithm", None)
    if algorithm_cfg is not None:
        algorithm_cfg.clip_param = 0.2
        algorithm_cfg.gamma = 0.99
        algorithm_cfg.lam = 0.95
        algorithm_cfg.learning_rate = 3e-4
        algorithm_cfg.num_learning_epochs = 5
        algorithm_cfg.num_mini_batches = 4
        algorithm_cfg.entropy_coef = 0.02
        algorithm_cfg.value_loss_coef = 1.0
        if hasattr(algorithm_cfg, "desired_kl"):
            algorithm_cfg.desired_kl = 0.01


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train BeamDojo Stage 1 with RSL-RL."""
    # make sure CLI overrides propagate
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    # enforce stage-specific defaults (before applying CLI overrides)
    _apply_stage1_defaults(agent_cfg)

    # environment overrides
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set seed/device overrides
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if getattr(args_cli, "device", None) else env_cfg.sim.device

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and getattr(args_cli, "device", None) and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print("=" * 80)
    print("BeamDojo Stage 1 Training")
    print("=" * 80)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    print(f"Target iterations: {agent_cfg.max_iterations}")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()

    print("\nStage 1 training complete!")
    print(f"Logs and checkpoints saved under: {log_dir}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
