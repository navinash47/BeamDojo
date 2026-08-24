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

    def test_stage2_resume_defaults_to_stage1_experiment(self):
        self.assertEqual(self.rt.resolve_load_experiment(2, "h1"), "beamdojo_h1_stage1")
        self.assertEqual(self.rt.resolve_load_experiment(2, "g1"), "beamdojo_g1_stage1")
        self.assertEqual(self.rt.resolve_load_experiment(1, "h1"), "beamdojo_h1_stage1")

    def test_load_experiment_override(self):
        self.assertEqual(
            self.rt.resolve_load_experiment(2, "h1", load_experiment="beamdojo_h1_stage2"),
            "beamdojo_h1_stage2",
        )
        with mock.patch.dict(os.environ, {"LOAD_EXPERIMENT": "beamdojo_h1_stage2"}, clear=False):
            self.assertEqual(self.rt.resolve_load_experiment(2, "h1"), "beamdojo_h1_stage2")

    def test_stage2_fine_tunes_stage1_flag(self):
        self.assertTrue(self.rt.stage2_fine_tunes_stage1(2, "h1"))
        self.assertTrue(self.rt.stage2_fine_tunes_stage1(2, "g1"))
        self.assertFalse(self.rt.stage2_fine_tunes_stage1(1, "h1"))
        self.assertFalse(
            self.rt.stage2_fine_tunes_stage1(2, "h1", load_experiment="beamdojo_h1_stage2")
        )

    def test_remaining_iters_stop_at_configured_max(self):
        self.assertEqual(self.rt.remaining_learning_iterations(0, 10_000), 10_000)
        self.assertEqual(self.rt.remaining_learning_iterations(500, 10_000), 9_500)
        self.assertEqual(self.rt.remaining_learning_iterations(10_000, 10_000), 0)
        self.assertEqual(self.rt.remaining_learning_iterations(None, 10_000), 10_000)

    def test_resume_picks_highest_iteration_not_missing_9999(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "2026-08-24_12-00-00"
            run.mkdir()
            (run / "model_0.pt").write_text("a")
            (run / "model_4.pt").write_text("b")
            (run / "model_10.pt").write_text("c")
            path = self.rt.resolve_resume_checkpoint(tmp)
        self.assertTrue(path.endswith("model_10.pt"))

    def test_resume_pinned_missing_9999_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "smoke"
            run.mkdir()
            (run / "model_4.pt").write_text("b")
            with self.assertRaises(ValueError):
                self.rt.resolve_resume_checkpoint(tmp, None, "model_9999.pt")

    def test_resume_picks_newer_run_by_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "2026-08-01_00-00-00"
            new = Path(tmp) / "2026-08-24_12-00-00"
            old.mkdir()
            new.mkdir()
            (old / "model_9999.pt").write_text("old")
            (new / "model_4.pt").write_text("new")
            os.utime(old, (1, 1))
            os.utime(new, (2_000_000_000, 2_000_000_000))
            path = self.rt.resolve_resume_checkpoint(tmp)
        self.assertTrue(path.endswith("model_4.pt"))
        self.assertIn("2026-08-24_12-00-00", path)

    def test_play_prefers_stage2_checkpoint_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            s1 = Path(tmp) / "rsl_rl" / "beamdojo_h1_stage1" / "run1"
            s2 = Path(tmp) / "rsl_rl" / "beamdojo_h1_stage2" / "run2"
            s1.mkdir(parents=True)
            s2.mkdir(parents=True)
            (s1 / "model_4.pt").write_text("s1")
            (s2 / "model_99.pt").write_text("s2")
            with mock.patch.dict(os.environ, {"BEAMDOJO_LOG_ROOT": tmp, "LOAD_EXPERIMENT": ""}, clear=False):
                path = self.rt.pick_play_checkpoint(2, "h1")
        self.assertTrue(path.endswith("model_99.pt"))

    def test_play_falls_back_to_stage1_when_stage2_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            s1 = Path(tmp) / "rsl_rl" / "beamdojo_h1_stage1" / "run1"
            s1.mkdir(parents=True)
            (s1 / "model_4.pt").write_text("s1")
            with mock.patch.dict(os.environ, {"BEAMDOJO_LOG_ROOT": tmp, "LOAD_EXPERIMENT": ""}, clear=False):
                path = self.rt.pick_play_checkpoint(2, "h1")
        self.assertTrue(path.endswith("model_4.pt"))
        self.assertIn("beamdojo_h1_stage1", path)


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

    def test_sync_wandb_copies_entity_to_username(self):
        with mock.patch.dict(os.environ, {"WANDB_ENTITY": "lab"}, clear=True):
            self.rt.sync_wandb_identity_env()
            self.assertEqual(os.environ["WANDB_USERNAME"], "lab")

    def test_sync_wandb_unsets_blank_username(self):
        with mock.patch.dict(os.environ, {"WANDB_USERNAME": "  "}, clear=True):
            self.rt.sync_wandb_identity_env()
            self.assertNotIn("WANDB_USERNAME", os.environ)

    def test_sync_wandb_prefers_entity_over_username(self):
        with mock.patch.dict(os.environ, {"WANDB_ENTITY": "team", "WANDB_USERNAME": "user"}, clear=True):
            self.rt.sync_wandb_identity_env()
            self.assertEqual(os.environ["WANDB_USERNAME"], "team")

    def test_sync_wandb_defaults_project(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.rt.sync_wandb_identity_env()
            self.assertEqual(os.environ["WANDB_PROJECT"], "beamdojo")

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

    def test_write_boot_status_is_unknown_not_running(self):
        rt = self.rt
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(rt, "REPO_ROOT", Path(tmp)):
                with mock.patch.dict(os.environ, {"WANDB_PROJECT": "beamdojo", "WANDB_API_KEY": "k"}, clear=True):
                    path = rt.write_boot_status(stage=1, robot="h1", terrain="beam")
            data = json.loads(path.read_text())
        self.assertEqual(data["status"], "unknown")
        self.assertEqual(data["task"], "Isaac-BeamDojo-Stage1-H1-v0")
        self.assertEqual(data["logger"], "wandb")
        self.assertIn("Not a live W&B run yet", data["note"])
        self.assertNotEqual(data["status"], "running")

    def test_clear_stale_distributed_env_unsets_world_size(self):
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "8", "RANK": "0", "LOCAL_RANK": "0"}, clear=False):
            cleared = self.rt.clear_stale_distributed_env(distributed=False)
            self.assertIn("WORLD_SIZE", cleared)
            self.assertNotIn("WORLD_SIZE", os.environ)
            self.assertNotIn("RANK", os.environ)

    def test_clear_stale_distributed_env_keeps_single_process(self):
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=False):
            self.assertEqual(self.rt.clear_stale_distributed_env(distributed=False), [])
            self.assertEqual(os.environ["WORLD_SIZE"], "1")

    def test_clear_stale_distributed_env_respects_flag(self):
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "1"}, clear=False):
            self.assertEqual(self.rt.clear_stale_distributed_env(distributed=True), [])
            self.assertEqual(os.environ["WORLD_SIZE"], "2")

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
        self.assertIn("W&B run URL", data["note"])

    def test_prepare_falls_back_when_wandb_init_fails(self):
        rt = self.rt
        fake_writer = object()

        class Runner:
            current_learning_iteration = 0
            writer = None
            log_dir = None
            logger_type = "wandb"

            def log(self, locs):
                return locs

            def _prepare_logging_writer(self):
                raise RuntimeError("wandb.init failed")

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(rt, "REPO_ROOT", Path(tmp)):
                with mock.patch.dict(os.environ, {"WANDB_PROJECT": "beamdojo"}, clear=True):
                    runner = Runner()
                    runner.log_dir = tmp
                    with mock.patch.object(rt, "_fallback_tensorboard_writer", return_value=fake_writer) as fallback:
                        rt.attach_status_heartbeat(runner, {"wandb_project": "beamdojo"}, every=10)
                        self.assertIs(runner._prepare_logging_writer(), fake_writer)
                        fallback.assert_called_once_with(runner)
                    data = json.loads((Path(tmp) / "tracking" / "training-status.json").read_text())
        self.assertEqual(data["status"], "running")
        self.assertIn("TensorBoard", data["note"])
        self.assertNotIn("/runs/", data.get("wandb_url") or "")

    def test_prepare_keeps_writer_when_post_init_fails(self):
        rt = self.rt
        existing = object()

        class Runner:
            current_learning_iteration = 0
            writer = existing
            log_dir = None

            def log(self, locs):
                return locs

            def _prepare_logging_writer(self):
                raise RuntimeError("store_config failed")

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(rt, "REPO_ROOT", Path(tmp)):
                with mock.patch.dict(os.environ, {"WANDB_PROJECT": "beamdojo"}, clear=True):
                    runner = Runner()
                    rt.attach_status_heartbeat(runner, {"wandb_project": "beamdojo"}, every=10)
                    with mock.patch.object(rt, "_fallback_tensorboard_writer") as fallback:
                        self.assertIs(runner._prepare_logging_writer(), existing)
                        fallback.assert_not_called()

    def test_mark_training_idle(self):
        rt = self.rt
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(rt, "REPO_ROOT", Path(tmp)):
                with mock.patch.dict(os.environ, {"WANDB_PROJECT": "beamdojo"}, clear=True):
                    path = rt.mark_training_idle("unit idle")
            data = json.loads(path.read_text())
        self.assertEqual(data["status"], "idle")
        self.assertEqual(data["note"], "unit idle")

    def test_status_write_skips_unwritable_extra_target(self):
        rt = self.rt
        with tempfile.TemporaryDirectory() as tmp:
            blocked = Path(tmp) / "blocked"
            blocked.write_text("not a directory")
            with mock.patch.object(rt, "REPO_ROOT", Path(tmp)):
                with mock.patch.dict(
                    os.environ,
                    {"WANDB_PROJECT": "beamdojo", "BEAMDOJO_LOG_ROOT": str(blocked)},
                    clear=True,
                ):
                    path = rt.write_training_status({"status": "idle", "note": "nfs skip"})
            data = json.loads(path.read_text())
        self.assertEqual(data["status"], "idle")
        self.assertEqual(data["note"], "nfs skip")

    def test_heartbeat_survives_status_write_error(self):
        rt = self.rt

        class Runner:
            current_learning_iteration = 10

            def log(self, locs):
                return locs

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(rt, "REPO_ROOT", Path(tmp)):
                with mock.patch.dict(os.environ, {"WANDB_PROJECT": "beamdojo"}, clear=True):
                    runner = Runner()
                    rt.attach_status_heartbeat(runner, {"wandb_project": "beamdojo"}, every=10)
                    with mock.patch.object(rt, "write_training_status", side_effect=OSError("nfs down")):
                        self.assertEqual(runner.log({"it": 10, "rewbuffer": [1.0]}), {"it": 10, "rewbuffer": [1.0]})

    def test_keep_wandb_call_swallows_http_errors(self):
        def boom(*_args, **_kwargs):
            raise RuntimeError("wandb.log 503")

        self.assertIsNone(self.rt._keep_wandb_call("add_scalar", boom, 1))

    def test_as_log_scalar_uses_item(self):
        class Tensorish:
            def item(self):
                return 0.25

        self.assertEqual(self.rt._as_log_scalar(Tensorish()), 0.25)
        self.assertEqual(self.rt._as_log_scalar(3.5), 3.5)

    def test_heartbeat_survives_log_error(self):
        rt = self.rt

        class Runner:
            current_learning_iteration = 10

            def log(self, locs):
                raise RuntimeError("add_scalar 503")

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(rt, "REPO_ROOT", Path(tmp)):
                with mock.patch.dict(os.environ, {"WANDB_PROJECT": "beamdojo", "WANDB_ENTITY": "x"}, clear=True):
                    runner = Runner()
                    rt.attach_status_heartbeat(runner, {"wandb_project": "beamdojo"}, every=10)
                    self.assertIsNone(runner.log({"it": 10, "rewbuffer": [2.0]}))
                    data = json.loads((Path(tmp) / "tracking" / "training-status.json").read_text())
        self.assertEqual(data["status"], "running")
        self.assertAlmostEqual(data["mean_reward"], 2.0)

    def test_save_survives_logger_upload_error(self):
        rt = self.rt

        class Runner:
            def save(self, path, infos=None):
                Path(path).write_bytes(b"ckpt")
                raise RuntimeError("wandb.save 503")

            def load(self, path, load_optimizer=True, map_location=None):
                return {"ok": True}

        rt._patch_runner_foot_optimizer(Runner)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model_0.pt")
            runner = Runner()
            runner.alg = type("Alg", (), {"foot_optimizer": None})()
            runner.save(path)
            self.assertTrue(Path(path).is_file())

    def test_save_survives_broken_foothold_splice(self):
        rt = self.rt

        class Foot:
            def state_dict(self):
                return {"x": 1}

        class Runner:
            def save(self, path, infos=None):
                Path(path).write_bytes(b"not-a-pickle")

            def load(self, path, load_optimizer=True, map_location=None):
                return None

        rt._patch_runner_foot_optimizer(Runner)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model_0.pt")
            runner = Runner()
            runner.alg = type("Alg", (), {"foot_optimizer": Foot()})()
            runner.save(path)
            self.assertEqual(Path(path).read_bytes(), b"not-a-pickle")

    def test_store_code_state_survives_dubious_ownership(self):
        class FakeOpr:
            pass

        def boom(*_a, **_k):
            raise RuntimeError("detected dubious ownership")

        FakeOpr.store_code_state = boom
        self.rt._patch_store_code_state(FakeOpr)
        self.assertEqual(FakeOpr.store_code_state("logs", []), [])

    def test_wandb_writer_patch_recovers_init_and_config_update(self):
        src = Path(__file__).resolve().parents[1] / "scripts" / "rsl_rl" / "beamdojo_runtime.py"
        text = src.read_text()
        self.assertIn("def _patch_wandb_config_update", text)
        self.assertIn("allow_val_change", text)
        self.assertIn("Keeping the W&B run if wandb.init already succeeded", text)
        self.assertIn('"log_config", orig_log_config', text)
        self.assertIn("def sanitize_ep_infos_for_rsl_log", text)
        self.assertIn("def _patch_runner_log_ep_infos", text)
        self.assertIn("def retry_call", text)
        self.assertIn("def _patch_wandb_init_retry", text)
        self.assertIn("def _apply_live_run_note", text)

    def test_retry_call_succeeds_after_transient_failures(self):
        n = {"i": 0}

        def flaky():
            n["i"] += 1
            if n["i"] < 3:
                raise RuntimeError("http 503")
            return "ok"

        slept = []
        self.assertEqual(
            self.rt.retry_call(flaky, attempts=3, label="wandb.init", sleeper=slept.append),
            "ok",
        )
        self.assertEqual(n["i"], 3)
        self.assertEqual(len(slept), 2)

    def test_retry_call_raises_after_attempts(self):
        def boom():
            raise RuntimeError("nope")

        with self.assertRaises(RuntimeError):
            self.rt.retry_call(boom, attempts=2, label="wandb.init", sleeper=lambda _s: None)

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
                return (None, None, None, None, {"log": {"Episode_Reward/foothold_penalty": 1.5}})

        env = Env()
        wrapped = rt.FootholdExtrasWrapper(env)
        info = wrapped.step(None)[-1]
        self.assertEqual(info["foothold_reward"], [-0.06, -0.02])
        self.assertEqual(info["log"]["Episode_Reward/foothold_penalty"], 1.5)
        self.assertAlmostEqual(info["log"]["foothold_penalty"], -0.04)

    def test_wrapper_copies_foothold_onto_unwrapped_extras(self):
        rt = self.rt

        class Vec(list):
            def __mul__(self, other):
                return Vec(x * other for x in self)

        class Env:
            unwrapped = None
            extras = {"log": {"Episode_Reward/foothold_penalty": 0.8}}
            beamdojo_foothold_step = Vec([-1.0])
            step_dt = 0.02

            def __init__(self):
                self.unwrapped = self

            def step(self, _action):
                return (None, None, None, None, {"log": {"Episode_Reward/foothold_penalty": 0.8}})

        env = Env()
        wrapped = rt.FootholdExtrasWrapper(env)
        info = wrapped.step(None)[-1]
        self.assertEqual(info["foothold_penalty"], [-0.02])
        self.assertEqual(env.extras["foothold_penalty"], [-0.02])
        self.assertEqual(env.extras["log"]["Episode_Reward/foothold_penalty"], 0.8)
        self.assertAlmostEqual(env.extras["log"]["foothold_penalty"], -0.02)
        self.assertAlmostEqual(info["log"]["foothold_penalty"], -0.02)

    def test_sanitize_ep_infos_collapses_per_env_and_drops_junk(self):
        infos = [
            {"Episode_Reward/x": 1.0, "foothold_penalty": [-0.4, -0.2], "note": "nope"},
            {"foothold_penalty": [-0.1]},
        ]
        self.rt.sanitize_ep_infos_for_rsl_log(infos)
        self.assertEqual(infos[0]["Episode_Reward/x"], 1.0)
        self.assertAlmostEqual(infos[0]["foothold_penalty"], -0.3)
        self.assertNotIn("note", infos[0])
        self.assertAlmostEqual(infos[1]["foothold_penalty"], -0.1)

    def test_patched_log_sanitizes_before_writer(self):
        seen = []

        class Runner:
            def log(self, locs, width=80, pad=35):
                del width, pad
                seen.append(locs["ep_infos"][0]["foothold_penalty"])

        self.rt._patch_runner_log_ep_infos(Runner)
        Runner().log({"ep_infos": [{"foothold_penalty": [-1.0, -3.0]}]})
        self.assertEqual(seen, [-2.0])

    def test_rsl_style_ep_info_cat_survives_sanitized_foothold(self):
        """rsl-rl 3.0.1 log() cats every extras['log'] key; mixed [N] vs 0-dim throws."""
        ep_infos = [
            {"Episode_Reward/track": 1.0, "foothold_penalty": [-0.06, -0.02]},
            {"Episode_Reward/track": 1.0, "foothold_penalty": [-0.04, 0.0]},
        ]
        self.rt.sanitize_ep_infos_for_rsl_log(ep_infos)
        for key in ep_infos[0]:
            infotensor = []
            for ep_info in ep_infos:
                value = ep_info[key]
                self.assertIsInstance(value, float)
                infotensor.append(value)
            self.assertEqual(len(infotensor), 2)


class RunnerCfgSanitizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rt = _load_runtime()

    def test_empty_hydra_rnd_cfg_is_sanitized_to_none(self):
        cfg = {
            "algorithm": {
                "class_name": "PPO",
                "rnd_cfg": {},
                "symmetry_cfg": {},
            }
        }
        self.rt.sanitize_rsl_rl_train_cfg(cfg)
        self.assertIsNone(cfg["algorithm"]["rnd_cfg"])
        self.assertIsNone(cfg["algorithm"]["symmetry_cfg"])

    def test_default_isaaclab_rnd_dump_is_sanitized_to_none(self):
        cfg = {
            "algorithm": {
                "rnd_cfg": {
                    "weight": 0.0,
                    "weight_schedule": None,
                    "learning_rate": 0.001,
                    "predictor_hidden_dims": [-1],
                    "target_hidden_dims": [-1],
                }
            }
        }
        self.rt.sanitize_rsl_rl_train_cfg(cfg)
        self.assertIsNone(cfg["algorithm"]["rnd_cfg"])

    def test_populated_rnd_cfg_is_left_intact(self):
        rnd = {"weight": 0.1, "learning_rate": 1e-4}
        cfg = {"algorithm": {"rnd_cfg": dict(rnd)}}
        self.rt.sanitize_rsl_rl_train_cfg(cfg)
        self.assertEqual(cfg["algorithm"]["rnd_cfg"], rnd)

    def test_runner_cfg_dict_sanitizes_to_dict_payload(self):
        class Agent:
            def to_dict(self):
                return {"algorithm": {"rnd_cfg": {}, "symmetry_cfg": {}}}

        out = self.rt.runner_cfg_dict(Agent())
        self.assertIsNone(out["algorithm"]["rnd_cfg"])
        self.assertIsNone(out["algorithm"]["symmetry_cfg"])


class ReassertGpuEnvCfgTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rt = _load_runtime()

    def test_parent_raycast_height_scan_detects_func_and_sensor(self):
        rt = self.rt

        class Func:
            __name__ = "height_scan"

        class Term:
            func = Func()
            params = {"sensor_cfg": {"name": "robot"}}

        self.assertTrue(rt.parent_raycast_height_scan(Term()))

        class TaskFunc:
            __name__ = "task_height_scan"

        class ParentSensor:
            name = "height_scanner"

        leftover = type("T", (), {"func": TaskFunc(), "params": {"sensor_cfg": ParentSensor()}})()
        self.assertTrue(rt.parent_raycast_height_scan(leftover))
        ok = type("T", (), {"func": TaskFunc(), "params": {"sensor_cfg": {"name": "robot"}}})()
        self.assertFalse(rt.parent_raycast_height_scan(ok))
        self.assertFalse(rt.parent_raycast_height_scan(None))

    def test_reassert_clears_scanner_commands_and_physics_material(self):
        plane_mat = object()
        cfg = type(
            "Cfg",
            (),
            {
                "scene": type(
                    "Scene",
                    (),
                    {
                        "height_scanner": object(),
                        "terrain": type(
                            "Terrain",
                            (),
                            {
                                "terrain_type": "plane",
                                "terrain_generator": None,
                                "debug_vis": False,
                                "physics_material": plane_mat,
                            },
                        )(),
                    },
                )(),
                "observations": type(
                    "Obs",
                    (),
                    {
                        "policy": type(
                            "Pol",
                            (),
                            {
                                "height_scan": type(
                                    "T",
                                    (),
                                    {
                                        "func": type("F", (), {"__name__": "task_height_scan"})(),
                                        "params": {"sensor_cfg": {"name": "robot"}},
                                    },
                                )()
                            },
                        )()
                    },
                )(),
                "commands": type(
                    "Cmd",
                    (),
                    {
                        "base_velocity": type(
                            "Vel",
                            (),
                            {"debug_vis": True, "heading_command": True},
                        )()
                    },
                )(),
                "sim": type("Sim", (), {"physics_material": object()})(),
            },
        )()
        self.rt.reassert_gpu_env_cfg(cfg)
        self.assertIsNone(cfg.scene.height_scanner)
        self.assertFalse(cfg.commands.base_velocity.debug_vis)
        self.assertFalse(cfg.commands.base_velocity.heading_command)
        self.assertIs(cfg.sim.physics_material, plane_mat)

    def test_reassert_flattens_generator_terrain(self):
        cfg = type(
            "Cfg",
            (),
            {
                "scene": type(
                    "Scene",
                    (),
                    {
                        "height_scanner": None,
                        "terrain": type(
                            "Terrain",
                            (),
                            {
                                "terrain_type": "generator",
                                "terrain_generator": object(),
                                "debug_vis": True,
                                "physics_material": "walk",
                                "visual_material": "nucleus-mdl",
                            },
                        )(),
                        "sky_light": type(
                            "Sky",
                            (),
                            {"spawn": type("Spawn", (), {"texture_file": "/nucleus/sky.hdr"})()},
                        )(),
                    },
                )(),
                "observations": type(
                    "Obs",
                    (),
                    {
                        "policy": type(
                            "Pol",
                            (),
                            {
                                "concatenate_terms": False,
                                "flatten_history_dim": False,
                                "height_scan": None,
                            },
                        )()
                    },
                )(),
                "commands": None,
                "sim": type("Sim", (), {"physics_material": "old"})(),
            },
        )()
        self.rt.reassert_gpu_env_cfg(cfg)
        self.assertEqual(cfg.scene.terrain.terrain_type, "plane")
        self.assertIsNone(cfg.scene.terrain.terrain_generator)
        self.assertFalse(cfg.scene.terrain.debug_vis)
        self.assertIsNone(cfg.scene.terrain.visual_material)
        self.assertIsNone(cfg.scene.sky_light.spawn.texture_file)
        self.assertTrue(cfg.observations.policy.concatenate_terms)
        self.assertTrue(cfg.observations.policy.flatten_history_dim)
        self.assertEqual(cfg.sim.physics_material, "walk")

    def test_reassert_replaces_parent_height_scan_without_isaac(self):
        called = []

        class Policy:
            height_scan = type(
                "T",
                (),
                {
                    "func": type("F", (), {"__name__": "height_scan"})(),
                    "params": {"sensor_cfg": {"name": "height_scanner"}},
                },
            )()

        cfg = type(
            "Cfg",
            (),
            {
                "scene": type("Scene", (), {"height_scanner": None, "terrain": None})(),
                "observations": type("Obs", (), {"policy": Policy()})(),
                "commands": None,
                "sim": None,
            },
        )()

        def fake_install(policy):
            called.append(policy)
            policy.height_scan = "task"

        with mock.patch.object(self.rt, "_install_task_height_scan", fake_install):
            self.rt.reassert_gpu_env_cfg(cfg)
        self.assertEqual(called, [cfg.observations.policy])
        self.assertEqual(cfg.observations.policy.height_scan, "task")

    def test_none_safe_from_dict_skips_none_target(self):
        seen = []

        def orig(obj, data, _ns=""):
            seen.append((obj, data, _ns))
            return "ok"

        wrapped = self.rt._none_safe_update_class_from_dict(orig)
        self.assertIsNone(wrapped(None, {"prim_path": "/World/ground"}, _ns="/scene/height_scanner"))
        self.assertEqual(seen, [])
        self.assertEqual(wrapped({"a": 1}, {"a": 2}, _ns="/x"), "ok")
        self.assertEqual(seen, [({"a": 1}, {"a": 2}, "/x")])


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
        train = (root / "scripts" / "rsl_rl" / "train_beamdojo.py").read_text()
        self.assertIn("resolve_load_log_root", train)
        self.assertIn("resolve_resume_checkpoint", train)
        self.assertIn("stage2_fine_tunes_stage1", train)
        self.assertIn("current_learning_iteration = 0", train)
        self.assertIn("remaining_learning_iterations", train)
        self.assertIn("Could not dump cfg yaml", train)
        self.assertIn("reassert_gpu_env_cfg(env_cfg)", train)
        self.assertLess(train.index("reassert_gpu_env_cfg(env_cfg)"), train.index("gym.make("))
        self.assertIn("Not a live W&B run yet", train)
        self.assertIn('"status": "unknown"', train)
        self.assertNotIn('{**status_body, "status": "running", "iteration": 0}', train)
        self.assertIn("write_boot_status", train)
        self.assertIn("clear_stale_distributed_env", train)
        self.assertLess(train.index("write_boot_status"), train.index("app_launcher = AppLauncher(args_cli)"))
        self.assertLess(
            train.index("clear_stale_distributed_env"),
            train.index("app_launcher = AppLauncher(args_cli)"),
        )
        runtime = (root / "scripts" / "rsl_rl" / "beamdojo_runtime.py").read_text()
        self.assertIn("def _patch_hydra_none_from_dict", runtime)
        self.assertIn("_patch_hydra_none_from_dict()", runtime)
        env_sh = (root / "scripts" / "cloud" / "_env.sh").read_text()
        self.assertIn("WANDB_USERNAME", env_sh)
        self.assertIn("WANDB_ENTITY", env_sh)
        self.assertIn('WANDB_PROJECT="${WANDB_PROJECT:-beamdojo}"', env_sh)
        self.assertIn("WANDB_INIT_TIMEOUT", env_sh)
        play = (root / "scripts" / "rsl_rl" / "play_beamdojo.py").read_text()
        self.assertIn("pick_play_checkpoint", play)
        self.assertIn("beamdojo_runtime.runner_cfg_dict(agent_cfg)", play)
        self.assertIn("reassert_gpu_env_cfg(env_cfg)", play)
        self.assertLess(play.index("reassert_gpu_env_cfg(env_cfg)"), play.index("gym.make("))
        self.assertIn("clear_stale_distributed_env", play)
        self.assertIn("beamdojo_runtime.runner_cfg_dict(agent_cfg)", train)
        self.assertNotIn("OnPolicyRunner(env, agent_cfg.to_dict()", train)
        self.assertNotIn("OnPolicyRunner(env, agent_cfg.to_dict()", play)
        stage2 = (root / "scripts" / "cloud" / "train_stage2.sh").read_text()
        self.assertIn("beamdojo_${ROBOT}_stage1", stage2)
        self.assertNotIn("model_9999.pt", stage2)
        self.assertNotIn("LOAD_RUN:?", stage2)
        self.assertIn("--resume", stage2)
        relaunch = (root / "scripts" / "cloud" / "after_relaunch.sh").read_text()
        self.assertIn("train_stage2.sh", relaunch)
        self.assertIn("BEAMDOJO_GIT_REF", relaunch)
        self.assertIn("return RigidObjectCfg(", relaunch)
        self.assertIn("sanitize_rsl_rl_train_cfg", relaunch)
        self.assertIn("_patch_store_code_state", relaunch)
        self.assertIn("sanitize_ep_infos_for_rsl_log", relaunch)
        self.assertIn("reassert_gpu_env_cfg", relaunch)
        self.assertIn("_patch_wandb_init_retry", relaunch)
        self.assertIn("write_boot_status", relaunch)
        self.assertIn("clear_stale_distributed_env", relaunch)
        self.assertIn('checkout -f -B "$REF" "origin/${REF}"', relaunch)
        self.assertIn("Never git clean", relaunch)
        self.assertNotIn("git clean", relaunch.replace("Never git clean", ""))
        self.assertIn("safe.directory", relaunch)
        self.assertIn("apply_physx_gpu_capacity", relaunch)
        self.assertIn("PHYSX_PATCH_COUNT_BEAM", relaunch)
        self.assertIn("gpu_max_rigid_contact_count", relaunch)
        self.assertIn("physx_gpu.py", relaunch)

    def test_stage2_catcher_and_ground_disable_are_wired(self):
        root = Path(__file__).resolve().parents[1]
        props = (root / "h1_cfg" / "scene_props.py").read_text()
        common = (root / "h1_cfg" / "beamdojo_common.py").read_text()
        mdp = (root / "h1_cfg" / "mdp.py").read_text()
        self.assertIn("def catcher_cfg", props)
        self.assertIn("CATCHER_Z", props)
        self.assertIn("collision_group=-1", props)
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
