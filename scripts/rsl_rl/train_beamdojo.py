#!/usr/bin/env python3
"""BeamDojo training entry point for RSL-RL (Stage 1/2, H1/G1, W&B)."""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import beamdojo_runtime  # isort: skip
import cli_args  # isort: skip


STAGE1_DEFAULT_SAVE_INTERVAL = 100
STAGE1_DEFAULT_MAX_ITERS = 10_000


parser = argparse.ArgumentParser(description="BeamDojo training with RSL-RL.")
parser.add_argument("--stage", type=int, choices=[1, 2], default=1, help="Stage 1 (imagined) or 2 (hard).")
parser.add_argument("--robot", type=str, choices=["h1", "g1"], default="h1")
parser.add_argument("--terrain", type=str, choices=["beam", "stones"], default="beam")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Gym registry ID. Default is derived from --stage/--robot/--terrain.",
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
    help="RL policy training iterations (defaults to 10k).",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

beamdojo_runtime.require_gpu_device(getattr(args_cli, "device", None))

if args_cli.task is None:
    args_cli.task = beamdojo_runtime.resolve_task(
        args_cli.stage, args_cli.robot, args_cli.terrain, play=False
    )

if args_cli.video:
    args_cli.enable_cameras = True

beamdojo_runtime.clear_stale_distributed_env(distributed=bool(getattr(args_cli, "distributed", False)))
_boot_payload = {
    "robot": args_cli.robot,
    "stage": args_cli.stage,
    "terrain": args_cli.terrain,
    "task": args_cli.task,
    "num_envs": int(args_cli.num_envs or 1024),
    "max_iterations": int(args_cli.max_iterations or STAGE1_DEFAULT_MAX_ITERS),
}
beamdojo_runtime.install_status_signal_hooks(_boot_payload)
_boot_path = beamdojo_runtime.write_boot_status(
    stage=args_cli.stage,
    robot=args_cli.robot,
    terrain=args_cli.terrain,
    task=args_cli.task,
    num_envs=_boot_payload["num_envs"],
    max_iterations=_boot_payload["max_iterations"],
    note=(
        "Starting Isaac Sim AppLauncher. gym.make / wandb.init come after this. "
        "Not a live W&B run yet."
    ),
)
print(f"[INFO] Wrote boot training-status (unknown) before Isaac starts: {_boot_path}")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab  # noqa: F401

beamdojo_runtime.ensure_beamdojo_registered()
beamdojo_runtime.inject_double_critic()

import importlib.metadata as metadata
import platform

from packaging import version

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

import os
from datetime import datetime

import gymnasium as gym
import torch

beamdojo_runtime.require_cuda()

import omni
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _apply_beamdojo_defaults(agent_cfg: RslRlBaseRunnerCfg):
    if getattr(args_cli, "experiment_name", None):
        agent_cfg.experiment_name = args_cli.experiment_name
    else:
        agent_cfg.experiment_name = beamdojo_runtime.experiment_name(args_cli.stage, args_cli.robot)
    if not getattr(agent_cfg, "run_name", None):
        agent_cfg.run_name = ""
    agent_cfg.save_interval = STAGE1_DEFAULT_SAVE_INTERVAL
    algorithm_cfg = getattr(agent_cfg, "algorithm", None)
    if algorithm_cfg is not None:
        algorithm_cfg.clip_param = 0.2
        algorithm_cfg.gamma = 0.99
        algorithm_cfg.lam = 0.95
        algorithm_cfg.learning_rate = 3e-4
        algorithm_cfg.num_learning_epochs = 5
        algorithm_cfg.num_mini_batches = 4
        if getattr(algorithm_cfg, "entropy_coef", None) is None:
            algorithm_cfg.entropy_coef = 0.02
        algorithm_cfg.value_loss_coef = 1.0
        if hasattr(algorithm_cfg, "desired_kl"):
            algorithm_cfg.desired_kl = 0.01
    beamdojo_runtime.apply_wandb_defaults(agent_cfg, args_cli)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train BeamDojo with RSL-RL."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    _apply_beamdojo_defaults(agent_cfg)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if getattr(args_cli, "device", None) else env_cfg.sim.device

    if args_cli.distributed and getattr(args_cli, "device", None) and "cpu" in args_cli.device:
        raise ValueError("Distributed training is not supported on CPU. Use --device cuda.")

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = beamdojo_runtime.resolve_log_root(agent_cfg.experiment_name)
    print("=" * 80)
    print(f"BeamDojo Stage {args_cli.stage} {args_cli.robot.upper()} ({args_cli.terrain})")
    print("=" * 80)
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    if getattr(agent_cfg, "logger", None) == "wandb":
        print(f"[INFO] Weights & Biases: {beamdojo_runtime.wandb_project_url(agent_cfg.wandb_project)}")
        print("[INFO] Open that URL on your Mac. Lambda has no public Isaac webpage.")
    else:
        print("[INFO] Logger is TensorBoard. Tunnel: ssh -L 6006:localhost:6006 lambda-beamdojo")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn("IO descriptors are only supported for manager based RL environments.")

    env_cfg.log_dir = log_dir

    status_body = {
        "robot": args_cli.robot,
        "stage": args_cli.stage,
        "terrain": args_cli.terrain,
        "task": args_cli.task,
        "num_envs": int(env_cfg.scene.num_envs),
        "num_steps_per_env": int(getattr(agent_cfg, "num_steps_per_env", 0) or 0),
        "max_iterations": int(agent_cfg.max_iterations),
        "logger": getattr(agent_cfg, "logger", "tensorboard"),
        "wandb_project": getattr(agent_cfg, "wandb_project", "beamdojo"),
        "wandb_url": beamdojo_runtime.live_wandb_url(getattr(agent_cfg, "wandb_project", "beamdojo")),
        "log_dir": log_dir,
        "checkpoint": None,
        "note": "Live curves are on W&B or TensorBoard. Checkpoints stay on NFS — do not git-commit .pt.",
    }
    beamdojo_runtime.install_status_signal_hooks(status_body)
    status_path = beamdojo_runtime.write_training_status(
        {
            **status_body,
            "status": "unknown",
            "iteration": 0,
            "note": (
                "Isaac Lab gym.make / runner init on CUDA. Not a live W&B run yet — "
                "status becomes running when learn() opens the logger."
            ),
        }
    )
    print(f"[INFO] Wrote training status: {status_path}")

    env = None
    runner = None
    resume_path = None
    try:
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            load_root = beamdojo_runtime.resolve_load_log_root(
                args_cli.stage,
                args_cli.robot,
                load_experiment=getattr(args_cli, "load_experiment", None),
            )
            print(f"[INFO] Resume checkpoints from: {load_root}")
            resume_path = beamdojo_runtime.resolve_resume_checkpoint(
                load_root,
                getattr(agent_cfg, "load_run", None),
                getattr(agent_cfg, "load_checkpoint", None),
            )
            print(f"[INFO] Resolved checkpoint: {resume_path}")
            status_body["checkpoint"] = resume_path

        beamdojo_runtime.reassert_gpu_env_cfg(env_cfg)
        beamdojo_runtime.reassert_agent_cuda(agent_cfg)
        beamdojo_runtime.reassert_clip_actions(agent_cfg)
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

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

        env = beamdojo_runtime.FootholdExtrasWrapper(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        beamdojo_runtime.reassert_runner_class(agent_cfg)
        train_cfg = beamdojo_runtime.runner_cfg_dict(agent_cfg)
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.add_git_repo_to_log(__file__)
        if resume_path:
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)
            if beamdojo_runtime.stage2_fine_tunes_stage1(
                args_cli.stage,
                args_cli.robot,
                load_experiment=getattr(args_cli, "load_experiment", None),
            ):
                runner.current_learning_iteration = 0
                print("[INFO] Stage 2 fine-tune: Stage 1 weights loaded, PPO iteration reset to 0.")

        try:
            dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
            dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        except Exception as dump_exc:
            print(f"[WARN] Could not dump cfg yaml ({type(dump_exc).__name__}): {dump_exc}")

        status_body["wandb_url"] = beamdojo_runtime.live_wandb_url(
            getattr(agent_cfg, "wandb_project", "beamdojo")
        )
        it0 = int(getattr(runner, "current_learning_iteration", 0) or 0)
        beamdojo_runtime.write_training_status(
            {
                **status_body,
                "status": "unknown",
                "iteration": it0,
                "note": (
                    "Runner constructed. learn() will call wandb.init next; "
                    "Research Lab flips to running when that heartbeat lands."
                ),
            }
        )
        beamdojo_runtime.attach_status_heartbeat(runner, status_body, every=10)

        remaining = beamdojo_runtime.remaining_learning_iterations(it0, agent_cfg.max_iterations)
        print(f"Target iterations: {agent_cfg.max_iterations} (starting at {it0}, remaining {remaining})")
        if remaining == 0:
            print("[INFO] Already at max_iterations; skipping learn().")
        else:
            runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)
        it = int(getattr(runner, "current_learning_iteration", 0) or 0)
        ckpt = os.path.join(log_dir, f"model_{it}.pt")
        beamdojo_runtime.mark_training_idle(
            f"Run finished. Copy {ckpt} off-box as insurance — never git-commit weights.",
            **{
                **status_body,
                "iteration": it,
                "wandb_url": beamdojo_runtime.live_wandb_url(
                    getattr(agent_cfg, "wandb_project", "beamdojo")
                ),
                "checkpoint": ckpt,
            },
        )
        print("\nTraining complete!")
        print(f"Logs and checkpoints saved under: {log_dir}")
    except Exception as exc:
        it = int(getattr(runner, "current_learning_iteration", 0) or 0) if runner is not None else 0
        beamdojo_runtime.mark_training_idle(
            f"Train stopped ({type(exc).__name__}). No live run. Terminate the A10 if it is idle.",
            **{**status_body, "iteration": it},
        )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
