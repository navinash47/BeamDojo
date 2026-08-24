"""Shared BeamDojo runtime helpers: GPU gate, env registration, NFS logs, W&B status."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = Path(__file__).resolve().parent
for _path in (REPO_ROOT, _SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from live_metrics import append_history, extract_live_metrics

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
    return f"beamdojo_{str(robot).lower()}_stage{int(stage)}"


def resolve_load_experiment(stage: int, robot: str, *, load_experiment: str | None = None) -> str:
    """Which ``logs/rsl_rl/<name>`` folder to read checkpoints from.

    Stage 2 fine-tunes Stage 1, so the default load experiment is Stage 1.
    Set ``LOAD_EXPERIMENT`` (or ``load_experiment``) to continue an interrupted
    Stage 2 run from ``beamdojo_<robot>_stage2``.
    """
    override = (load_experiment or os.environ.get("LOAD_EXPERIMENT") or "").strip()
    if override:
        return override
    if int(stage) >= 2:
        return experiment_name(1, robot)
    return experiment_name(stage, robot)


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
    _patch_runner_foot_optimizer(opr.OnPolicyRunner)


def _patch_runner_foot_optimizer(runner_cls) -> None:
    """Persist the foothold Adam state next to rsl-rl's loco optimizer."""
    if getattr(runner_cls, "_beamdojo_foot_ckpt", False):
        return
    orig_save = runner_cls.save
    orig_load = runner_cls.load

    def save(self, path, infos=None):
        orig_save(self, path, infos)
        foot = getattr(getattr(self, "alg", None), "foot_optimizer", None)
        if foot is None:
            return
        import torch

        blob = torch.load(path, map_location="cpu", weights_only=False)
        blob["foot_optimizer_state_dict"] = foot.state_dict()
        torch.save(blob, path)

    def load(self, path, load_optimizer=True, map_location=None):
        infos = orig_load(self, path, load_optimizer=load_optimizer, map_location=map_location)
        foot = getattr(getattr(self, "alg", None), "foot_optimizer", None)
        if not load_optimizer or foot is None:
            return infos
        import torch

        blob = torch.load(path, map_location=map_location, weights_only=False)
        state = blob.get("foot_optimizer_state_dict")
        if not state:
            return infos
        try:
            foot.load_state_dict(state)
        except Exception as exc:
            print(f"[WARN] Not loading foothold optimizer: {exc}")
        return infos

    runner_cls.save = save
    runner_cls.load = load
    runner_cls._beamdojo_foot_ckpt = True


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


def resolve_load_log_root(
    stage: int,
    robot: str,
    *,
    load_experiment: str | None = None,
) -> str:
    """Log root used by ``get_checkpoint_path`` when resuming."""
    return resolve_log_root(resolve_load_experiment(stage, robot, load_experiment=load_experiment))


def wandb_project_url(project: str = "beamdojo") -> str:
    entity = (os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_USERNAME") or "").strip()
    if entity:
        return f"https://wandb.ai/{entity}/{project}"
    return "https://wandb.ai"


def live_wandb_identity(project: str = "beamdojo") -> tuple[str, str | None, str]:
    """Active W&B run URL/entity/project once wandb.init has run; else env fallbacks."""
    entity = (os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_USERNAME") or "").strip() or None
    proj = project
    url = None
    try:
        import wandb

        run = getattr(wandb, "run", None)
        if run is not None:
            url = getattr(run, "url", None)
            if not url and hasattr(run, "get_url"):
                url = run.get_url()
            ent = getattr(run, "entity", None)
            pr = getattr(run, "project", None)
            if ent:
                entity = str(ent)
            if pr:
                proj = str(pr)
    except Exception:
        pass
    if url:
        return str(url), entity, proj
    return wandb_project_url(proj), entity, proj


def live_wandb_url(project: str = "beamdojo") -> str:
    """Prefer the active W&B run URL once wandb.init has run; else the project page."""
    url, _, _ = live_wandb_identity(project)
    return url


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
    project = str(payload.get("wandb_project") or os.environ.get("WANDB_PROJECT", "beamdojo"))
    url, entity, project = live_wandb_identity(project)
    body = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "unknown",
        "host": "lambda-a10" if Path("/lambda/nfs/beamdojo").is_dir() else "local",
        "wandb_project": project,
        "wandb_entity": entity,
        "wandb_url": url,
        **payload,
    }
    # A real wandb.run URL always wins over a stale project homepage in payload.
    if url and "/runs/" in url:
        body["wandb_url"] = url
        body["wandb_project"] = project
        if entity:
            body["wandb_entity"] = entity
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


def _status_from_runner(runner, payload: dict, *, iteration: int | None = None) -> dict:
    project = payload.get("wandb_project") or os.environ.get("WANDB_PROJECT", "beamdojo")
    url, entity, project = live_wandb_identity(project)
    it = int(iteration if iteration is not None else getattr(runner, "current_learning_iteration", 0) or 0)
    log_dir = payload.get("log_dir")
    ckpt = None
    if log_dir:
        candidate = os.path.join(str(log_dir), f"model_{it}.pt")
        if os.path.isfile(candidate):
            ckpt = candidate
    payload["wandb_url"] = url
    payload["wandb_entity"] = entity
    payload["wandb_project"] = project
    return {
        **payload,
        "status": "running",
        "iteration": it,
        "wandb_url": url,
        "wandb_entity": entity,
        "wandb_project": project,
        "checkpoint": ckpt or payload.get("checkpoint"),
    }


def attach_status_heartbeat(runner, payload: dict, *, every: int = 10) -> None:
    """Rewrite gitignored training-status.json when W&B inits and every N PPO iters.

    OnPolicyRunner._prepare_logging_writer runs wandb.init at the start of learn(),
    which is the first moment a real run URL exists. log() is called once per
    iteration after that. Kingdom syncs this file into the Research Lab.
    """
    orig_prepare = getattr(runner, "_prepare_logging_writer", None)
    if callable(orig_prepare):

        def _prepare(*args, **kwargs):
            result = orig_prepare(*args, **kwargs)
            write_training_status(_status_from_runner(runner, payload))
            return result

        runner._prepare_logging_writer = _prepare

    orig_log = getattr(runner, "log", None)
    if not callable(orig_log):
        return

    def _log(*args, **kwargs):
        result = orig_log(*args, **kwargs)
        it = int(getattr(runner, "current_learning_iteration", 0) or 0)
        if every > 0 and it % every != 0:
            return result
        locs = args[0] if args else kwargs.get("locs")
        metrics = extract_live_metrics(
            locs if isinstance(locs, dict) else {},
            num_envs=int(payload.get("num_envs") or 0),
            num_steps=int(
                getattr(runner, "num_steps_per_env", 0) or payload.get("num_steps_per_env") or 0
            ),
        )
        payload.update(metrics)
        append_history(payload, metrics, it)
        write_training_status(_status_from_runner(runner, payload, iteration=it))
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
