"""Isaac-free contracts for H1/G1 env cfgs used on the A10 path."""

from __future__ import annotations

import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (_REPO / rel).read_text()


class EnvCfgContractTests(unittest.TestCase):
    def test_h1_spawns_minimal_usd(self):
        common = _read("h1_cfg/beamdojo_common.py")
        self.assertIn("H1_MINIMAL_CFG", common)
        self.assertIn("G1_MINIMAL_CFG", common)

    def test_beamdojo_env_base_declares_hydra_fields(self):
        src = _read("h1_cfg/beamdojo_env_base.py")
        for needle in (
            "class BeamDojoSceneCfg",
            "class BeamDojoRewardsCfg",
            "class BeamDojoEventCfg",
            "class BeamDojoTerminationsCfg",
            "class BeamDojoCurriculumCfg",
            "class BeamDojoEnvCfg",
            "foothold_penalty",
            "task_beam",
            "catcher",
            "task_stone_0",
            "task_stone_23",
            "joint_deviation_fingers",
            "disable_ground",
            "off_terrain",
            "beam_width",
            "init_beamdojo",
        ):
            self.assertIn(needle, src)

    def test_stage_cfgs_inherit_beamdojo_env(self):
        for rel in (
            "h1_cfg/beamdojo_stage1_cfg.py",
            "h1_cfg/beamdojo_stage2_cfg.py",
            "g1_cfg/beamdojo_stage1_cfg.py",
            "g1_cfg/beamdojo_stage2_cfg.py",
        ):
            src = _read(rel)
            self.assertIn("BeamDojoEnvCfg", src)
            self.assertNotIn("LocomotionVelocityRoughEnvCfg", src)

    def test_g1_gets_official_finger_and_leg_filters(self):
        common = _read("h1_cfg/beamdojo_common.py")
        self.assertIn("G1_FINGER_JOINTS", common)
        self.assertIn(".*_five_joint", common)
        self.assertIn("joint_deviation_fingers", common)
        self.assertIn(".*_hip_.*", common)
        self.assertIn(".*_knee_joint", common)
        self.assertIn(".*_ankle_.*", common)

    def test_stage2_reset_stays_on_beam(self):
        common = _read("h1_cfg/beamdojo_common.py")
        self.assertIn("Spawn on the beam", common)
        self.assertIn('position_range"] = (1.0, 1.0)', common)
        self.assertIn("def _zero_root_reset_velocity", common)
        self.assertIn("heading_command = False", common)
        self.assertIn("debug_vis = False", common)
        shared = common.split("def apply_stage1")[0]
        self.assertIn("heading_command = False", shared)
        self.assertIn("cfg.scene.height_scanner = None", common)
        self.assertNotIn("RayCasterCfg", common)

    def test_stone_count_matches_declared_slots(self):
        props = _read("h1_cfg/scene_props.py")
        self.assertIn("STONE_COUNT = 24", props)
        self.assertIn("count: int = STONE_COUNT", props)


if __name__ == "__main__":
    unittest.main()
