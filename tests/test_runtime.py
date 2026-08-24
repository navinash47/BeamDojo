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

    def test_g1_stage2_stones(self):
        self.assertEqual(
            self.rt.resolve_task(2, "g1", "stones"),
            "Isaac-BeamDojo-Stage2-G1-Stones-v0",
        )

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

    def test_live_url_prefers_wandb_run(self):
        fake = mock.MagicMock()
        fake.run = mock.MagicMock()
        fake.run.url = "https://wandb.ai/avinash/beamdojo/runs/abc123"
        with mock.patch.dict("sys.modules", {"wandb": fake}):
            self.assertEqual(
                self.rt.live_wandb_url("beamdojo"),
                "https://wandb.ai/avinash/beamdojo/runs/abc123",
            )

    def test_live_url_falls_back_without_run(self):
        with mock.patch.dict(os.environ, {"WANDB_ENTITY": "lab"}, clear=True):
            self.assertEqual(self.rt.live_wandb_url("beamdojo"), "https://wandb.ai/lab/beamdojo")

    def test_live_identity_reads_entity_from_run(self):
        fake = mock.MagicMock()
        fake.run = mock.MagicMock()
        fake.run.url = "https://wandb.ai/fromrun/beamdojo/runs/xyz"
        fake.run.entity = "fromrun"
        fake.run.project = "beamdojo"
        with mock.patch.dict("sys.modules", {"wandb": fake}):
            url, entity, project = self.rt.live_wandb_identity("other")
        self.assertEqual(url, "https://wandb.ai/fromrun/beamdojo/runs/xyz")
        self.assertEqual(entity, "fromrun")
        self.assertEqual(project, "beamdojo")


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

    def test_heartbeat_writes_every_n_iters(self):
        rt = self.rt
        logs = []

        class Runner:
            current_learning_iteration = 0

            def log(self, locs):
                logs.append(locs)

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(rt, "REPO_ROOT", Path(tmp)):
                with mock.patch.dict(os.environ, {"WANDB_PROJECT": "beamdojo", "WANDB_ENTITY": "x"}, clear=True):
                    runner = Runner()
                    rt.attach_status_heartbeat(
                        runner,
                        {"wandb_project": "beamdojo", "log_dir": str(Path(tmp) / "run")},
                        every=10,
                    )
                    runner.current_learning_iteration = 3
                    runner.log({"it": 3})
                    self.assertFalse((Path(tmp) / "tracking" / "training-status.json").exists())
                    runner.current_learning_iteration = 10
                    runner.log({"it": 10})
                    data = json.loads((Path(tmp) / "tracking" / "training-status.json").read_text())
        self.assertEqual(logs, [{"it": 3}, {"it": 10}])
        self.assertEqual(data["status"], "running")
        self.assertEqual(data["iteration"], 10)
        self.assertEqual(data["wandb_url"], "https://wandb.ai/x/beamdojo")

    def test_heartbeat_writes_live_metrics(self):
        rt = self.rt
        from collections import deque

        class Runner:
            current_learning_iteration = 10
            num_steps_per_env = 24

            def log(self, locs):
                return locs

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(rt, "REPO_ROOT", Path(tmp)):
                with mock.patch.dict(os.environ, {"WANDB_PROJECT": "beamdojo", "WANDB_ENTITY": "x"}, clear=True):
                    runner = Runner()
                    payload = {"wandb_project": "beamdojo", "num_envs": 64, "num_steps_per_env": 24}
                    rt.attach_status_heartbeat(runner, payload, every=10)
                    runner.log(
                        {
                            "rewbuffer": deque([1.0, 3.0, 5.0]),
                            "lenbuffer": deque([8.0, 12.0]),
                            "loss_dict": {"value_function": 0.4, "foothold_value_function": 0.2},
                            "collection_time": 1.0,
                            "learn_time": 1.0,
                            "collection_size": 200,
                        }
                    )
                    data = json.loads((Path(tmp) / "tracking" / "training-status.json").read_text())
        self.assertEqual(data["status"], "running")
        self.assertAlmostEqual(data["mean_reward"], 3.0)
        self.assertAlmostEqual(data["mean_episode_length"], 10.0)
        self.assertAlmostEqual(data["value_loss"], 0.4)
        self.assertAlmostEqual(data["foothold_value_loss"], 0.2)
        self.assertAlmostEqual(data["fps"], 100.0)
        self.assertEqual(data["history"][-1]["iteration"], 10)
        self.assertAlmostEqual(data["history"][-1]["mean_reward"], 3.0)
        self.assertNotIn("rewbuffer", data)

    def test_prepare_logging_writer_writes_run_url(self):
        rt = self.rt
        fake = mock.MagicMock()
        fake.run = mock.MagicMock()
        fake.run.url = "https://wandb.ai/x/beamdojo/runs/live1"
        fake.run.entity = "x"
        fake.run.project = "beamdojo"

        class Runner:
            current_learning_iteration = 0

            def log(self, locs):
                return locs

            def _prepare_logging_writer(self):
                return "writer"

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(rt, "REPO_ROOT", Path(tmp)):
                with mock.patch.dict(os.environ, {"WANDB_PROJECT": "beamdojo"}, clear=True):
                    with mock.patch.dict("sys.modules", {"wandb": fake}):
                        runner = Runner()
                        rt.attach_status_heartbeat(runner, {"wandb_project": "beamdojo"}, every=10)
                        self.assertEqual(runner._prepare_logging_writer(), "writer")
                        data = json.loads((Path(tmp) / "tracking" / "training-status.json").read_text())
        self.assertEqual(data["status"], "running")
        self.assertEqual(data["wandb_url"], "https://wandb.ai/x/beamdojo/runs/live1")
        self.assertEqual(data["wandb_entity"], "x")

    def test_mark_training_idle(self):
        rt = self.rt
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(rt, "REPO_ROOT", Path(tmp)):
                with mock.patch.dict(os.environ, {"WANDB_PROJECT": "beamdojo"}, clear=True):
                    path = rt.mark_training_idle("unit idle")
            data = json.loads(path.read_text())
        self.assertEqual(data["status"], "idle")
        self.assertEqual(data["note"], "unit idle")

    def test_wrapper_writes_log_and_top_level_foothold(self):
        rt = self.rt

        class Vec(list):
            def __mul__(self, other):
                return Vec(x * other for x in self)

        class Env:
            unwrapped = None
            beamdojo_foothold_step = Vec([-3.0, -1.0])
            step_dt = 0.02

            def __init__(self):
                self.unwrapped = self

            def step(self, _action):
                return (None, None, None, None, {"log": {}})

        env = Env()
        wrapped = rt.FootholdExtrasWrapper(env)
        info = wrapped.step(None)[-1]
        self.assertEqual(info["foothold_reward"], [-0.06, -0.02])
        self.assertEqual(info["log"]["foothold_penalty"], [-0.06, -0.02])


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
            "Isaac-BeamDojo-Stage2-G1-Stones-v0",
        ]:
            self.assertIn(gym_id, text)
        g1s2 = (root / "g1_cfg" / "beamdojo_stage2_cfg.py").read_text()
        self.assertIn("BeamDojoG1Stage2PPORunnerCfg", g1s2)

    def test_stage2_catcher_and_ground_disable_are_wired(self):
        root = Path(__file__).resolve().parents[1]
        props = (root / "h1_cfg" / "scene_props.py").read_text()
        common = (root / "h1_cfg" / "beamdojo_common.py").read_text()
        mdp = (root / "h1_cfg" / "mdp.py").read_text()
        self.assertIn("def catcher_cfg", props)
        self.assertIn("CATCHER_Z", props)
        self.assertIn("cfg.scene.catcher = catcher_cfg()", common)
        self.assertIn("disable_ground_collision", common)
        self.assertIn("def disable_ground_collision", mdp)
        self.assertNotIn("No-op helper kept for wrappers", mdp)

    def test_paper_table_ix_dr_is_wired(self):
        common = (Path(__file__).resolve().parents[1] / "h1_cfg" / "beamdojo_common.py").read_text()
        self.assertIn("def apply_paper_dr", common)
        self.assertIn('mass_distribution_params"] = (-2.0, 2.0)', common)
        self.assertIn("static_friction_range", common)
        self.assertIn("apply_paper_dr(cfg, spec)", common)
        self.assertIn("torso = spec.torso_body", common)
        # Training no longer zeros payload/CoM DR.
        self.assertNotIn("cfg.events.add_base_mass = None\n    cfg.events.base_com = None\n    cfg.events.push_robot = None", common.split("def apply_play")[0])

    def test_g1_spec_matches_isaaclab_g1_minimal_names(self):
        from h1_cfg.robot_spec import G1

        self.assertEqual(G1.torso_body, "torso_link")
        self.assertEqual(G1.torso_joint, "torso_joint")
        self.assertEqual(G1.feet_body, ".*_ankle_roll_link")
        self.assertEqual(len(G1.action_joints or []), 6)
        self.assertIn(".*_hip_pitch_joint", G1.action_joints)
        self.assertIn(".*_ankle_roll_joint", G1.action_joints)

    def test_stage2_base_contact_uses_robot_torso_body(self):
        spec = (Path(__file__).resolve().parents[1] / "h1_cfg" / "robot_spec.py").read_text()
        common = (Path(__file__).resolve().parents[1] / "h1_cfg" / "beamdojo_common.py").read_text()
        self.assertIn("torso_body", spec)
        self.assertIn("spec.torso_body", common)


if __name__ == "__main__":
    unittest.main()
