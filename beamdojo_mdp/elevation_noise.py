"""Paper appendix VI-C elevation-map measurement noise (Isaac-free).

Applied to the *scan* only, not to the foothold reward (true task map).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from beamdojo_mdp.heightfield import BEAM_OFF_Z, BEAM_ON_Z, _is_torch


def tilt_height_grid(hz, grid_n: int, hx, hy):
    """Roll/pitch as a bilinear height ramp (paper: interpolate ±hx, ±hy)."""
    if _is_torch(hz):
        import torch

        xs = torch.linspace(-1.0, 1.0, grid_n, device=hz.device, dtype=hz.dtype)
        xx, yy = torch.meshgrid(xs, xs, indexing="ij")
        ramp = xx.reshape(-1) * hx.reshape(-1, 1) + yy.reshape(-1) * hy.reshape(-1, 1)
        return hz + ramp
    xs = np.linspace(-1.0, 1.0, grid_n, dtype=np.asarray(hz).dtype)
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    hx_b = np.asarray(hx).reshape(-1, 1)
    hy_b = np.asarray(hy).reshape(-1, 1)
    ramp = xx.reshape(1, -1) * hx_b + yy.reshape(1, -1) * hy_b
    return np.asarray(hz) + ramp


def dilate_on_cells(hz, grid_n: int, on_z: float = BEAM_ON_Z, off_z: float = BEAM_OFF_Z, extend=None):
    """Foothold extension: 4-neighbor dilate of on-beam cells (LiDAR smoothing)."""
    mid = 0.5 * (float(on_z) + float(off_z))
    if _is_torch(hz):
        import torch

        grid = hz.reshape(-1, grid_n, grid_n)
        on = grid > mid
        pad = torch.nn.functional.pad(on.float(), (1, 1, 1, 1))
        neigh = torch.maximum(
            torch.maximum(pad[:, 1:-1, 1:-1], pad[:, :-2, 1:-1]),
            torch.maximum(pad[:, 2:, 1:-1], torch.maximum(pad[:, 1:-1, :-2], pad[:, 1:-1, 2:])),
        ).bool()
        dilated = torch.where(neigh, torch.as_tensor(on_z, device=hz.device, dtype=hz.dtype), grid)
        if extend is None:
            return dilated.reshape(hz.shape)
        mask = extend.reshape(-1).bool()
        out = torch.where(mask.view(-1, 1, 1), dilated, grid)
        return out.reshape(hz.shape)
    grid = np.asarray(hz).reshape(-1, grid_n, grid_n)
    on = grid > mid
    pad = np.pad(on, ((0, 0), (1, 1), (1, 1)))
    neigh = (
        pad[:, 1:-1, 1:-1]
        | pad[:, :-2, 1:-1]
        | pad[:, 2:, 1:-1]
        | pad[:, 1:-1, :-2]
        | pad[:, 1:-1, 2:]
    )
    dilated = np.where(neigh, on_z, grid)
    if extend is None:
        return dilated.reshape(np.asarray(hz).shape)
    mask = np.asarray(extend).reshape(-1).astype(bool)
    out = np.where(mask[:, None, None], dilated, grid)
    return out.reshape(np.asarray(hz).shape)


def maybe_repeat_map(current, previous, repeat_mask) -> Any:
    """Map-repeat (latency): keep the previous scan where ``repeat_mask`` is true."""
    if previous is None:
        return current
    if _is_torch(current):
        import torch

        mask = repeat_mask.reshape(-1).bool().view(-1, *([1] * (current.ndim - 1)))
        return torch.where(mask, previous, current)
    mask = np.asarray(repeat_mask).reshape(-1).astype(bool)
    shape = (mask.shape[0],) + (1,) * (np.asarray(current).ndim - 1)
    return np.where(mask.reshape(shape), previous, current)
