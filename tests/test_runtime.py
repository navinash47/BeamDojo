"""Tests for gym ids, task routing, W&B URL, and training-status writer."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REPO = Path(__file__).resolve().parents[1]


def _load_runtime():
    path = _REPO / "scripts" / "rsl_rl" / "beamdojo_runtime.py"
    spec = importlib.util.spec_from_file_location("beamdojo_runtime_under_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TaskRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rt = _load_runtime()

    def test_stage1_h1(self):
        self.assertEqual(self.rt.resolve_task(1, "h1"), "Isaac-BeamDojo-Stage1-H1-v0")

    def test_stage2_h1_stones_play(self):
        self.assertEqual(
            self.rt.resolve_task(2, "h1", "stones", play=True),
            "Isaac-BeamDojo-Stage2-H1-Stones-Play-v0",
        )

    def test_g1_stage2(self):
        self.assertEqual(self.rt.resolve_task(2, "g1"), "Isaac-BeamDojo-Stage2-G1-v0")

    def test_unknown_combo(self):
        with self.assertRaises(ValueError):
            self.rt.resolve_task(1, "h1", "stones")

    def test_experiment_name(self):
        self.assertEqual(self.rt.experiment_name(2, "g1"), "beamdojo_g1_stage2")


class WandbUrlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rt = _load_runtime()

    def test_anonymous_project(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.rt.wandb_project_url("beamdojo"), "https://wandb.ai")

    def test_entity_project(self):
        with mock.patch.dict(os.environ, {"WANDB_ENTITY": "avinash"}, clear=True):
            self.assertEqual(
                self.rt.wandb_project_url("beamdojo"),
                "https://wandb.ai/avinash/beamdojo",
            )


class TrainingStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rt = _load_runtime()

    def test_write_roundtrip(self):
        rt = self.rt
        with tempfile.TemporaryDirectory() as tmp:
            tracking = Path(tmp) / "tracking"
            tracking.mkdir()
            with mock.patch.object(rt, "REPO_ROOT", Path(tmp)):
                with mock.patch.dict(os.environ, {"WANDB_PROJECT": "beamdojo", "WANDB_ENTITY": "x"}, clear=True):
                    path = rt.write_training_status({"status": "idle", "note": "unit test"})
            data = json.loads(path.read_text())
            self.assertEqual(data["status"], "idle")
            self.assertEqual(data["wandb_url"], "https://wandb.ai/x/beamdojo")
            self.assertIn("updated", data)


class GymIdSourceTests(unittest.TestCase):
    def test_cfg_files_register_expected_ids(self):
        root = Path(__file__).resolve().parents[1]
        text = ""
        for rel in [
            "h1_cfg/beamdojo_stage1_cfg.py",
            "h1_cfg/beamdojo_stage2_cfg.py",
            "g1_cfg/beamdojo_stage1_cfg.py",
            "g1_cfg/beamdojo_stage2_cfg.py",
        ]:
            text += (root / rel).read_text()
        for gym_id in [
            "Isaac-BeamDojo-Stage1-H1-v0",
            "Isaac-BeamDojo-Stage2-H1-v0",
            "Isaac-BeamDojo-Stage2-H1-Stones-v0",
            "Isaac-BeamDojo-Stage1-G1-v0",
            "Isaac-BeamDojo-Stage2-G1-v0",
        ]:
            self.assertIn(gym_id, text)


if __name__ == "__main__":
    unittest.main()
