"""Stage 2 beam-width curriculum helpers (Isaac-free math)."""

from __future__ import annotations


def physical_beam_scale_y(current_width: float, spawn_width: float = 0.40) -> float:
    """Y scale so a cuboid spawned at ``spawn_width`` matches ``current_width``."""
    return float(current_width) / max(float(spawn_width), 1e-6)


def should_update_physical_width(prev: float | None, current: float, min_delta: float = 0.005) -> bool:
    if prev is None:
        return True
    return abs(float(current) - float(prev)) >= float(min_delta)


def task_beam_prim_paths(env_prim_paths=None, num_envs: int = 0) -> list[str]:
    if env_prim_paths:
        return [f"{str(root).rstrip('/')}/TaskBeam" for root in env_prim_paths]
    return [f"/World/envs/env_{i}/TaskBeam" for i in range(int(num_envs))]
