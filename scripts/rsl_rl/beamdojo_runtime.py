"""Shared BeamDojo runtime helpers: GPU gate, env registration, NFS logs, W&B status."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_REGISTERED = False

TASK_IDS = {
    (1, "h1", "beam"): "Isaac-BeamDojo-Stage1-H1-v0",
    (2, "h1", "beam"): "Isaac-BeamDojo-Stage2-H1-v0",
    (2, "h1", "stones"): "Isaac-BeamDojo-Stage2-H1-Stones-v0",
    (1, "g1", "beam"): "Isaac-BeamDojo-Stage1-G1-v0",
    (2, "g1", "beam"): "Isaac-BeamDojo-Stage2-G1-v0",
    (2, "g1", "stones"): "Isaac-BeamDojo-Stage2-G1-Stones-v0",
}

PLAY_IDS = {
    (1, "h1", "beam"): "Isaac-BeamDojo-Stage1-H1-Play-v0",
    (2, "h1", "beam"): "Isaac-BeamDojo-Stage2-H1-Play-v0",
    (2, "h1", "stones"): "Isaac-BeamDojo-Stage2-H1-Stones-Play-v0",
    (1, "g1", "beam"): "Isaac-BeamDojo-Stage1-G1-Play-v0",
    (2, "g1", "beam"): "Isaac-BeamDojo-Stage2-G1-Play-v0",
    (2, "g1", "stones"): "Isaac-BeamDojo-Stage2-G1-Stones-Play-v0",
}


def require_gpu_device(device: str | None) -> None:
    """Refuse CPU before Isaac Sim starts. Visualization and training must be CUDA."""
    if device is None:
        return
    if str(device).strip().lower().startswith("cpu"):
        raise SystemExit(
            "BeamDojo refuses CPU. Pass --device cuda:0 on an NVIDIA GPU with RT cores (A10/A6000/L40S)."
        )


def require_cuda() -> None:
    """Abort if PyTorch cannot see CUDA. Call after torch is imported."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. BeamDojo will not train or render on CPU. "
            "Run on the Lambda A10 (or another RT-core NVIDIA GPU)."
        )
    if torch.cuda.device_count() < 1:
        raise SystemExit("No CUDA devices found. Refusing to continue.")


def resolve_task(stage: int, robot: str, terrain: str = "beam", *, play: bool = False) -> str:
    table = PLAY_IDS if play else TASK_IDS
    key = (int(stage), str(robot).lower(), str(terrain).lower())
    if key not in table:
        raise ValueError(
            f"No gym id for stage={stage} robot={robot} terrain={terrain} play={play}. "
            f"Known: {sorted(table)}"
        )
    return table[key]


def experiment_name(stage: int, robot: str) -> str:
    return f"beamdojo_{robot}_stage{stage}"


def ensure_beamdojo_registered() -> None:
    """Import cfg modules so gym.register side effects run. Requires SimulationApp."""
    global _REGISTERED
    if _REGISTERED:
        return
    import g1_cfg.beamdojo_stage1_cfg  # noqa: F401
    import g1_cfg.beamdojo_stage2_cfg  # noqa: F401
    import h1_cfg.beamdojo_stage1_cfg  # noqa: F401
    import h1_cfg.beamdojo_stage2_cfg  # noqa: F401

    _REGISTERED = True


def ensure_beamdojo_stage1_registered() -> None:
    ensure_beamdojo_registered()


def inject_double_critic() -> None:
    """Put ActorCriticDouble / PPODoubleCritic in rsl-rl's eval() namespace."""
    import rsl_rl.runners.on_policy_runner as opr
    from beamdojo_agents.double_critic import ActorCriticDouble, PPODoubleCritic

    opr.ActorCriticDouble = ActorCriticDouble
    opr.PPODoubleCritic = PPODoubleCritic


def resolve_log_root(experiment_name: str) -> str:
    """Prefer Lambda NFS so checkpoints survive instance terminate."""
    env_root = os.environ.get("BEAMDOJO_LOG_ROOT", "").strip()
    nfs = Path("/lambda/nfs/beamdojo/logs")
    if env_root:
        base = Path(env_root)
    elif nfs.is_dir():
        base = nfs
    else:
        base = REPO_ROOT / "logs"
    path = (base / "rsl_rl" / experiment_name).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def wandb_project_url(project: str = "beamdojo") -> str:
    entity = (os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_USERNAME") or "").strip()
    if entity:
        return f"https://wandb.ai/{entity}/{project}"
    return "https://wandb.ai"


def live_wandb_url(project: str = "beamdojo") -> str:
    """Prefer the active W&B run URL once wandb.init has run; else the project page."""
    try:
        import wandb

        run = getattr(wandb, "run", None)
        if run is not None:
            url = getattr(run, "url", None)
            if not url and hasattr(run, "get_url"):
                url = run.get_url()
            if url:
                return str(url)
    except Exception:
        pass
    return wandb_project_url(project)


def apply_wandb_defaults(agent_cfg, args_cli) -> None:
    """Use W&B when a key is present unless the user picked another logger."""
    logger = getattr(args_cli, "logger", None)
    if logger:
        agent_cfg.logger = logger
    elif os.environ.get("WANDB_API_KEY", "").strip():
        agent_cfg.logger = "wandb"
    project = getattr(args_cli, "log_project_name", None) or os.environ.get("WANDB_PROJECT", "beamdojo")
    if getattr(agent_cfg, "logger", None) in {"wandb", "neptune"}:
        agent_cfg.wandb_project = project
        agent_cfg.neptune_project = project


def write_training_status(payload: dict) -> Path:
    """Write live-run JSON for Kingdom Research Lab (gitignored)."""
    body = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "unknown",
        "host": "lambda-a10" if Path("/lambda/nfs/beamdojo").is_dir() else "local",
        "wandb_project": os.environ.get("WANDB_PROJECT", "beamdojo"),
        "wandb_entity": os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_USERNAME") or None,
        "wandb_url": live_wandb_url(os.environ.get("WANDB_PROJECT", "beamdojo")),
        **payload,
    }
    raw = json.dumps(body, indent=2) + "\n"
    targets = [REPO_ROOT / "tracking" / "training-status.json"]
    env_root = os.environ.get("BEAMDOJO_LOG_ROOT", "").strip()
    if env_root:
        targets.append(Path(env_root) / "training-status.json")
    nfs = Path("/lambda/nfs/beamdojo/logs/training-status.json")
    if nfs.parent.is_dir():
        targets.append(nfs)
    written = targets[0]
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw)
        written = path
    return written


def attach_status_heartbeat(runner, payload: dict, *, every: int = 10) -> None:
    """Rewrite gitignored training-status.json every ``every`` PPO iterations.

    OnPolicyRunner.log is called once per iteration. Kingdom syncs this file into
    the Research Lab; W&B remains the live metric webpage.
    """
    orig_log = getattr(runner, "log", None)
    if not callable(orig_log):
        return

    def _log(*args, **kwargs):
        result = orig_log(*args, **kwargs)
        it = int(getattr(runner, "current_learning_iteration", 0) or 0)
        if every > 0 and it % every != 0:
            return result
        project = payload.get("wandb_project") or os.environ.get("WANDB_PROJECT", "beamdojo")
        log_dir = payload.get("log_dir")
        ckpt = None
        if log_dir:
            candidate = os.path.join(str(log_dir), f"model_{it}.pt")
            if os.path.isfile(candidate):
                ckpt = candidate
        write_training_status(
            {
                **payload,
                "status": "running",
                "iteration": it,
                "wandb_url": live_wandb_url(project),
                "checkpoint": ckpt or payload.get("checkpoint"),
            }
        )
        return result

    runner.log = _log


def mark_training_idle(note: str | None = None, **payload) -> Path:
    """Force idle so Kingdom never keeps a dead A10 marked running."""
    return write_training_status(
        {
            **payload,
            "status": "idle",
            "note": note
            or "Idle — GPU instance terminating or no train running. Checkpoints stay on NFS.",
        }
    )


def install_status_signal_hooks(payload: dict) -> None:
    """On SIGINT/SIGTERM, write idle before exit (Lambda terminate / Ctrl-C)."""
    import signal

    def _handle(signum, _frame):
        mark_training_idle(
            f"Caught signal {signum}; marking idle so Research Lab does not show a live train.",
            **payload,
        )
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


class FootholdExtrasWrapper:
    """Gym wrapper: put per-step foothold term into extras for the double critic."""

    def __init__(self, env):
        self.env = env
        self.unwrapped = getattr(env, "unwrapped", env)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        result = self.env.step(action)
        raw = self.unwrapped
        foot = getattr(raw, "beamdojo_foothold_step", None)
        if foot is None:
            return result
        dt = float(getattr(raw, "step_dt", 0.02) or 0.02)
        contrib = foot * dt
        info = result[-1]
        if isinstance(info, dict):
            info["foothold_reward"] = contrib
            info["foothold_penalty"] = contrib
            log = info.get("log")
            if not isinstance(log, dict):
                log = {}
                info["log"] = log
            log["foothold_reward"] = contrib
            log["foothold_penalty"] = contrib
        return result

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def close(self):
        return self.env.close()
