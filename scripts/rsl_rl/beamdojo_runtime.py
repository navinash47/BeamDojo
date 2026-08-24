"""Shared BeamDojo runtime helpers: GPU gate, env registration, NFS logs."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def ensure_beamdojo_stage1_registered() -> None:
    """Load Stage 1 gym ids from this repo, not from a copy inside Isaac Lab."""
    module_name = "beamdojo_stage1_cfg"
    if module_name in sys.modules:
        return

    cfg_path = REPO_ROOT / "h1_cfg" / "beamdojo_stage1_cfg.py"
    spec = importlib.util.spec_from_file_location(module_name, cfg_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load BeamDojo Stage 1 config from {cfg_path}")

    module = importlib.util.module_from_spec(spec)
    module.agents = importlib.import_module(
        "isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents"
    )
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


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
