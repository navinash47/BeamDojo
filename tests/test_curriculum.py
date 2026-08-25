"""Isaac-free tests for Stage 2 physical beam curriculum."""

from __future__ import annotations

import unittest
from pathlib import Path

from beamdojo_mdp.curriculum import (
    physical_beam_scale_y,
    should_update_physical_width,
    task_beam_prim_paths,
)


class CurriculumTests(unittest.TestCase):
    def test_scale_halves_when_width_halves(self):
        self.assertAlmostEqual(physical_beam_scale_y(0.20, 0.40), 0.5)

    def test_skip_tiny_width_changes(self):
        self.assertTrue(should_update_physical_width(None, 0.40))
        self.assertFalse(should_update_physical_width(0.40, 0.401))
        self.assertTrue(should_update_physical_width(0.40, 0.39))

    def test_prim_paths_from_scene_roots(self):
        self.assertEqual(
            task_beam_prim_paths(["/World/envs/env_0", "/World/envs/env_1/"], 2),
            ["/World/envs/env_0/TaskBeam", "/World/envs/env_1/TaskBeam"],
        )

    def test_mdp_wires_physical_scale(self):
        text = (Path(__file__).resolve().parents[1] / "h1_cfg" / "mdp.py").read_text()
        self.assertIn("_scale_task_beam_prims", text)
        self.assertIn("physical_beam_scale_y", text)


if __name__ == "__main__":
    unittest.main()
