"""Dual-terrain heightfield: physics stays flat; the policy sees this map.

Stage 1 (paper §IV-C.1): walk on a plane, scan/reward against an imagined beam
or stepping-stone field. Stage 2 uses the same query with real collision.
"""

from __future__ import annotations

import math
from typing import Any

BEAM_ON_Z = 0.0
BEAM_OFF_Z = -0.40


def _is_torch(x: Any) -> bool:
    return type(x).__module__.startswith("torch")


def beam_surface_z(
    x,
    y,
    origin_x,
    origin_y,
    length: float,
    width,
    y_center: float = 0.0,
    on_z: float = BEAM_ON_Z,
    off_z: float = BEAM_OFF_Z,
    x_start: float = 0.0,
):
    """Height of the task beam at world XY. `width` may be a scalar or per-env tensor."""
    xe = x - origin_x
    ye = y - origin_y
    half = width * 0.5
    on = (xe >= x_start) & (xe <= (x_start + length)) & (ye >= (y_center - half)) & (ye <= (y_center + half))
    if _is_torch(on):
        import torch

        on_v = torch.as_tensor(on_z, device=x.device, dtype=x.dtype)
        off_v = torch.as_tensor(off_z, device=x.device, dtype=x.dtype)
        return torch.where(on, on_v, off_v)
    import numpy as np

    return np.where(on, on_z, off_z)


def stone_surface_z(
    x,
    y,
    origin_x,
    origin_y,
    length: float,
    stone_size: float,
    gap: float,
    y_center: float = 0.0,
    on_z: float = BEAM_ON_Z,
    off_z: float = BEAM_OFF_Z,
    corridor_width: float | None = None,
):
    """Stepping stones along +X: `stone_size` pads separated by `gap`."""
    xe = x - origin_x
    ye = y - origin_y
    pitch = stone_size + gap
    if _is_torch(xe):
        import torch

        safe_pitch = max(float(pitch), 1e-6)
        cell = torch.floor(xe / safe_pitch)
        local = xe - cell * safe_pitch
        on_x = (xe >= 0) & (xe <= length) & (local < stone_size)
        half = (corridor_width if corridor_width is not None else stone_size) * 0.5
        on_y = (ye >= (y_center - half)) & (ye <= (y_center + half))
        on = on_x & on_y
        on_v = torch.as_tensor(on_z, device=x.device, dtype=x.dtype)
        off_v = torch.as_tensor(off_z, device=x.device, dtype=x.dtype)
        return torch.where(on, on_v, off_v)
    import numpy as np

    safe_pitch = max(float(pitch), 1e-6)
    cell = np.floor(xe / safe_pitch)
    local = xe - cell * safe_pitch
    on_x = (xe >= 0) & (xe <= length) & (local < stone_size)
    half = (corridor_width if corridor_width is not None else stone_size) * 0.5
    on_y = (ye >= (y_center - half)) & (ye <= (y_center + half))
    return np.where(on_x & on_y, on_z, off_z)


def yaw_grid_xy(px, py, yaw, n: int = 15, extent: float = 1.4):
    """n×n XY grid in the robot yaw frame, returned as (x, y) with trailing sample dim."""
    if _is_torch(px):
        import torch

        xs = torch.linspace(-extent / 2, extent / 2, n, device=px.device, dtype=px.dtype)
        xx, yy = torch.meshgrid(xs, xs, indexing="ij")
        local = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        rot_x = local[:, 0] * c.unsqueeze(-1) - local[:, 1] * s.unsqueeze(-1)
        rot_y = local[:, 0] * s.unsqueeze(-1) + local[:, 1] * c.unsqueeze(-1)
        return px.unsqueeze(-1) + rot_x, py.unsqueeze(-1) + rot_y
    import numpy as np

    xs = np.linspace(-extent / 2, extent / 2, n)
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    local = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    c = np.cos(yaw)[..., None]
    s = np.sin(yaw)[..., None]
    rot_x = local[:, 0] * c - local[:, 1] * s
    rot_y = local[:, 0] * s + local[:, 1] * c
    return px[..., None] + rot_x, py[..., None] + rot_y


def foothold_off_count(
    sample_x,
    sample_y,
    origin_x,
    origin_y,
    length: float,
    width,
    depth_threshold: float = -0.1,
    on_z: float = BEAM_ON_Z,
    off_z: float = BEAM_OFF_Z,
    sample_z=None,
    terrain: str = "beam",
    stone_size: float = 0.20,
    gap: float = 0.10,
):
    """Count samples whose clearance vs the task map is below `depth_threshold` (paper eq. 2)."""
    if terrain == "stones":
        hz = stone_surface_z(
            sample_x,
            sample_y,
            origin_x,
            origin_y,
            length,
            stone_size,
            gap,
            on_z=on_z,
            off_z=off_z,
        )
    else:
        hz = beam_surface_z(
            sample_x,
            sample_y,
            origin_x,
            origin_y,
            length,
            width,
            on_z=on_z,
            off_z=off_z,
        )
    if sample_z is None:
        clearance = hz - on_z
    else:
        clearance = sample_z - hz
    off = clearance < depth_threshold
    if _is_torch(off):
        return off.to(dtype=sample_x.dtype).sum(dim=-1)
    return off.astype(float).sum(axis=-1)


def yaw_from_quat_wxyz(quat):
    """Extract yaw from (w, x, y, z) quaternions."""
    if _is_torch(quat):
        import torch

        w, x, y, z = quat.unbind(-1)
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    import numpy as np

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
