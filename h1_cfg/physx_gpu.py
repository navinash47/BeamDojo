"""A10-safe PhysX GPU buffer floors (Isaac Lab 2.3.2).

Isaac Lab ``PhysxCfg`` (v2.3.2) preallocates these in pinned host / GPU memory;
they cannot grow at runtime:

* rigid contact stream default = 8_388_608
* rigid patch stream default = 163_840; locomotion parent raises that to
  ``10 * 2**15`` (327_680)
* found/lost pair capacity default = 2_097_152

Dual-terrain adds cloned kinematic cuboids (one beam, or 24 stones × 1024
envs). Undersized *patch* buffers abort PhysX before ``learn()`` / W&B.
Oversized floors — especially doubling the 8M contact stream — OOM an A10
24GB the same way: Isaac Sim + 1024 H1 already occupy most of the card.

Do not raise the contact-stream default. Only bump patch count, and only bump
found/lost pairs for the 24-stone scene.
"""

from __future__ import annotations

# Modest bump over locomotion's 10 * 2**15. One extra kinematic box per env.
PHYSX_PATCH_COUNT_BEAM = 16 * 2**15  # 524_288

# 24 stones × 1024 envs. Still well below a 2M patch buffer.
PHYSX_PATCH_COUNT_STONES = 2**20  # 1_048_576

# Isaac 2.3.2 default is already 2**21; stones need more pair-discovery room.
PHYSX_FOUND_LOST_STONES = 2**22  # 4_194_304

# Anything at or above this is an A10 24GB OOM waiting to happen.
PHYSX_A10_UNSAFE_FLOOR = 2**23  # 8_388_608; Isaac contact-stream default


def physx_gpu_floors(*, stones: bool = False) -> dict[str, int]:
    """Name → minimum PhysX GPU buffer. Contact stream is intentionally absent."""
    floors = {"gpu_max_rigid_patch_count": PHYSX_PATCH_COUNT_BEAM}
    if stones:
        floors["gpu_max_rigid_patch_count"] = PHYSX_PATCH_COUNT_STONES
        floors["gpu_found_lost_pairs_capacity"] = PHYSX_FOUND_LOST_STONES
    return floors


def apply_physx_gpu_floors(physx, *, stones: bool = False) -> None:
    if physx is None:
        return
    for name, floor in physx_gpu_floors(stones=stones).items():
        if floor >= PHYSX_A10_UNSAFE_FLOOR:
            raise ValueError(
                f"PhysX floor {name}={floor} is at/above the A10-unsafe threshold "
                f"{PHYSX_A10_UNSAFE_FLOOR}. Dual-terrain only needs a patch bump."
            )
        current = getattr(physx, name, None)
        try:
            value = int(current or 0)
        except (TypeError, ValueError):
            value = 0
        setattr(physx, name, max(value, floor))


def apply_physx_gpu_capacity(cfg, *, stones: bool = False) -> None:
    """Raise dual-terrain PhysX GPU buffers without doubling the contact stream."""
    physx = getattr(getattr(cfg, "sim", None), "physx", None)
    apply_physx_gpu_floors(physx, stones=stones)
