"""Isaac-free PhysX GPU floor checks so dual-terrain can start on A10 24GB."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from h1_cfg.physx_gpu import (
    PHYSX_A10_UNSAFE_FLOOR,
    PHYSX_FOUND_LOST_STONES,
    PHYSX_PATCH_COUNT_BEAM,
    PHYSX_PATCH_COUNT_STONES,
    apply_physx_gpu_capacity,
    physx_gpu_floors,
)


class PhysxGpuFloorTests(unittest.TestCase):
    def test_beam_only_bumps_patch_count(self):
        floors = physx_gpu_floors(stones=False)
        self.assertEqual(set(floors), {"gpu_max_rigid_patch_count"})
        self.assertEqual(floors["gpu_max_rigid_patch_count"], 16 * 2**15)
        self.assertEqual(PHYSX_PATCH_COUNT_BEAM, 16 * 2**15)
        self.assertLess(PHYSX_PATCH_COUNT_BEAM, 2**20)
        self.assertLess(PHYSX_PATCH_COUNT_BEAM, PHYSX_A10_UNSAFE_FLOOR)

    def test_stones_add_found_lost_not_contact_stream(self):
        floors = physx_gpu_floors(stones=True)
        self.assertEqual(
            set(floors),
            {"gpu_max_rigid_patch_count", "gpu_found_lost_pairs_capacity"},
        )
        self.assertEqual(floors["gpu_max_rigid_patch_count"], PHYSX_PATCH_COUNT_STONES)
        self.assertEqual(floors["gpu_found_lost_pairs_capacity"], PHYSX_FOUND_LOST_STONES)
        self.assertEqual(PHYSX_PATCH_COUNT_STONES, 2**20)
        self.assertLess(PHYSX_PATCH_COUNT_STONES, PHYSX_A10_UNSAFE_FLOOR)
        self.assertLess(PHYSX_FOUND_LOST_STONES, PHYSX_A10_UNSAFE_FLOOR)

    def test_apply_does_not_raise_isaac_contact_stream(self):
        physx = SimpleNamespace(
            gpu_max_rigid_patch_count=10 * 2**15,
            gpu_max_rigid_contact_count=2**23,
            gpu_found_lost_pairs_capacity=2**21,
            gpu_total_aggregate_pairs_capacity=2**21,
        )
        cfg = SimpleNamespace(sim=SimpleNamespace(physx=physx))
        apply_physx_gpu_capacity(cfg, stones=False)
        self.assertEqual(physx.gpu_max_rigid_patch_count, PHYSX_PATCH_COUNT_BEAM)
        self.assertEqual(physx.gpu_max_rigid_contact_count, 2**23)
        self.assertEqual(physx.gpu_found_lost_pairs_capacity, 2**21)
        self.assertEqual(physx.gpu_total_aggregate_pairs_capacity, 2**21)

        apply_physx_gpu_capacity(cfg, stones=True)
        self.assertEqual(physx.gpu_max_rigid_patch_count, PHYSX_PATCH_COUNT_STONES)
        self.assertEqual(physx.gpu_max_rigid_contact_count, 2**23)
        self.assertEqual(physx.gpu_found_lost_pairs_capacity, PHYSX_FOUND_LOST_STONES)
        self.assertEqual(physx.gpu_total_aggregate_pairs_capacity, 2**21)

    def test_missing_physx_is_a_no_op(self):
        apply_physx_gpu_capacity(SimpleNamespace(), stones=False)
        apply_physx_gpu_capacity(SimpleNamespace(sim=SimpleNamespace()), stones=True)


if __name__ == "__main__":
    unittest.main()
