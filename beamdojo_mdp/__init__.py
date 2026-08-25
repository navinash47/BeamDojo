"""Isaac-free BeamDojo task geometry (imagined / physical beam + stones)."""

from .heightfield import (
    BEAM_OFF_Z,
    BEAM_ON_Z,
    beam_surface_z,
    foothold_off_count,
    stone_surface_z,
    yaw_grid_xy,
)

__all__ = [
    "BEAM_OFF_Z",
    "BEAM_ON_Z",
    "beam_surface_z",
    "foothold_off_count",
    "stone_surface_z",
    "yaw_grid_xy",
]
