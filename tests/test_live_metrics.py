"""Tests for JSON-safe PPO metrics extracted from OnPolicyRunner.log(locals())."""

from __future__ import annotations

import importlib.util
import unittest
from collections import deque
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _load():
    path = _REPO / "scripts" / "rsl_rl" / "live_metrics.py"
    spec = importlib.util.spec_from_file_location("live_metrics_under_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LiveMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_extracts_reward_length_losses_and_fps(self):
        metrics = self.m.extract_live_metrics(
            {
                "rewbuffer": deque([1.0, 3.0, 5.0]),
                "lenbuffer": deque([10.0, 20.0]),
                "loss_dict": {
                    "value_function": 0.5,
                    "surrogate": 0.1,
                    "entropy": 0.02,
                    "foothold_value_function": 0.3,
                },
                "collection_time": 1.0,
                "learn_time": 1.0,
                "collection_size": 240,
            }
        )
        self.assertAlmostEqual(metrics["mean_reward"], 3.0)
        self.assertAlmostEqual(metrics["mean_episode_length"], 15.0)
        self.assertAlmostEqual(metrics["value_loss"], 0.5)
        self.assertAlmostEqual(metrics["surrogate_loss"], 0.1)
        self.assertAlmostEqual(metrics["entropy"], 0.02)
        self.assertAlmostEqual(metrics["foothold_value_loss"], 0.3)
        self.assertAlmostEqual(metrics["fps"], 120.0)

    def test_empty_buffers_are_omitted(self):
        metrics = self.m.extract_live_metrics({"rewbuffer": deque(), "lenbuffer": []})
        self.assertEqual(metrics, {})

    def test_reads_double_critic_value_foothold_key(self):
        metrics = self.m.extract_live_metrics({"loss_dict": {"value_foothold": 0.22}})
        self.assertAlmostEqual(metrics["foothold_value_loss"], 0.22)

    def test_skips_non_finite_and_non_scalars(self):
        metrics = self.m.extract_live_metrics(
            {
                "rewbuffer": deque([float("nan"), "nope", 4.0]),
                "loss_dict": {"value_function": object()},
            }
        )
        self.assertAlmostEqual(metrics["mean_reward"], 4.0)
        self.assertNotIn("value_loss", metrics)

    def test_fps_falls_back_to_env_step_product(self):
        metrics = self.m.extract_live_metrics(
            {"collection_time": 2.0, "learn_time": 2.0},
            num_envs=64,
            num_steps=24,
        )
        self.assertAlmostEqual(metrics["fps"], 384.0)

    def test_history_caps_and_skips_empty_metrics(self):
        payload: dict = {}
        self.m.append_history(payload, {}, 1)
        self.assertNotIn("history", payload)
        for i in range(130):
            self.m.append_history(payload, {"mean_reward": float(i)}, i)
        self.assertEqual(len(payload["history"]), 120)
        self.assertEqual(payload["history"][0]["iteration"], 10)
        self.assertEqual(payload["history"][-1]["mean_reward"], 129.0)


if __name__ == "__main__":
    unittest.main()
