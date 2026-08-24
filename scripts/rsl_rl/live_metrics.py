"""Scalar training metrics for tracking/training-status.json (JSON-safe only)."""

from __future__ import annotations

from typing import Any

HISTORY_CAP = 120


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _mean(values: Any) -> float | None:
    if values is None:
        return None
    try:
        items = list(values)
    except TypeError:
        return _finite(values)
    if not items:
        return None
    total = 0.0
    count = 0
    for item in items:
        number = _finite(item)
        if number is None:
            continue
        total += number
        count += 1
    if count == 0:
        return None
    return total / count


def _loss_scalars(loss_dict: Any) -> dict[str, float]:
    if not isinstance(loss_dict, dict):
        return {}
    mapping = {
        "value_loss": ("value_function", "value_loss"),
        "surrogate_loss": ("surrogate", "surrogate_loss"),
        "entropy": ("entropy",),
        # PPODoubleCritic.update() writes value_foothold; rsl-rl PPO uses value_function.
        "foothold_value_loss": ("value_foothold", "foothold_value_function", "foothold_value_loss"),
    }
    out: dict[str, float] = {}
    for dest, sources in mapping.items():
        for src in sources:
            number = _finite(loss_dict.get(src))
            if number is not None:
                out[dest] = number
                break
    return out


def extract_live_metrics(locs: dict[str, Any] | None, *, num_envs: int = 0, num_steps: int = 0) -> dict[str, float]:
    """Pull JSON-safe scalars from OnPolicyRunner.log(locals())."""
    if not isinstance(locs, dict):
        return {}
    metrics: dict[str, float] = {}
    reward = _mean(locs.get("rewbuffer"))
    if reward is not None:
        metrics["mean_reward"] = reward
    length = _mean(locs.get("lenbuffer"))
    if length is not None:
        metrics["mean_episode_length"] = length
    metrics.update(_loss_scalars(locs.get("loss_dict")))
    collection = _finite(locs.get("collection_time"))
    learn = _finite(locs.get("learn_time"))
    if collection is not None:
        metrics["collection_time"] = collection
    if learn is not None:
        metrics["learn_time"] = learn
    steps = int(locs.get("collection_size") or 0)
    if steps <= 0 and num_envs > 0 and num_steps > 0:
        steps = int(num_envs) * int(num_steps)
    duration = 0.0
    if collection is not None:
        duration += collection
    if learn is not None:
        duration += learn
    if steps > 0 and duration > 0:
        metrics["fps"] = steps / duration
    return metrics


def append_history(payload: dict[str, Any], metrics: dict[str, float], iteration: int) -> None:
    point: dict[str, float | int] = {"iteration": int(iteration)}
    for key in ("mean_reward", "mean_episode_length", "fps"):
        if key in metrics:
            point[key] = metrics[key]
    if len(point) <= 1:
        return
    history = list(payload.get("history") or [])
    history.append(point)
    payload["history"] = history[-HISTORY_CAP:]
