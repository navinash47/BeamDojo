"""Isaac-free checks that the double critic module is importable for GPU inject."""

from __future__ import annotations

import inspect
import unittest

from beamdojo_mdp.advantage import W1, W2


class DoubleCriticModuleTests(unittest.TestCase):
    def test_imports_without_nameerror(self):
        import beamdojo_agents.double_critic as dc

        sig = inspect.signature(dc.PPODoubleCritic.__init__)
        self.assertEqual(sig.parameters["w1"].default, W1)
        self.assertEqual(sig.parameters["w2"].default, W2)

    def test_foothold_update_runs_before_parent_update(self):
        import beamdojo_agents.double_critic as dc

        src = inspect.getsource(dc.PPODoubleCritic.update)
        self.assertIn("_update_foothold_critic", src)
        self.assertLess(src.find("_update_foothold_critic"), src.find("super().update"))

    def test_process_env_step_bootstraps_timeouts_after_split(self):
        import beamdojo_agents.double_critic as dc

        src = inspect.getsource(dc.PPODoubleCritic.process_env_step)
        split_at = src.find("loco = rewards - foot")
        boot_at = src.find("_timeout_bootstrap_reward")
        super_at = src.find("super().process_env_step")
        self.assertGreater(split_at, -1)
        self.assertGreater(boot_at, split_at)
        self.assertGreater(super_at, boot_at)
        self.assertIn("time_outs", inspect.getsource(dc._timeout_bootstrap_reward))

    def test_missing_foothold_checkpoint_skips_optimizer_resume(self):
        import beamdojo_agents.double_critic as dc

        src = inspect.getsource(dc.ActorCriticDouble.load_state_dict)
        self.assertIn("critic_foothold", src)
        self.assertIn("return False", src)
        self.assertLess(src.find("return False"), src.rfind("return True"))

    def test_foothold_optimizer_clears_grads_before_loco_update(self):
        import beamdojo_agents.double_critic as dc

        src = inspect.getsource(dc.PPODoubleCritic._update_foothold_critic)
        step_at = src.find("self.foot_optimizer.step()")
        clear_at = src.rfind("self.foot_optimizer.zero_grad")
        self.assertGreater(step_at, -1)
        self.assertGreater(clear_at, step_at)
        self.assertIn("set_to_none=True", src[clear_at:])

    def test_unknown_ppo_kwargs_are_dropped(self):
        import beamdojo_agents.double_critic as dc

        class FakePPO:
            def __init__(self, policy, gamma=0.99, lam=0.95):
                del policy, gamma, lam

        previous = dc.PPO
        dc.PPO = FakePPO
        try:
            out = dc._ppo_init_kwargs({"gamma": 0.9, "class_name": "PPO", "lam": 0.8})
        finally:
            dc.PPO = previous
        self.assertEqual(out, {"gamma": 0.9, "lam": 0.8})

    def test_default_rnd_dict_becomes_none(self):
        import beamdojo_agents.double_critic as dc

        class FakePPO:
            def __init__(self, policy, gamma=0.99, lam=0.95, rnd_cfg=None, symmetry_cfg=None):
                del policy, gamma, lam, rnd_cfg, symmetry_cfg

        previous = dc.PPO
        dc.PPO = FakePPO
        try:
            out = dc._ppo_init_kwargs(
                {
                    "gamma": 0.99,
                    "lam": 0.95,
                    "rnd_cfg": {"weight": 0.0, "learning_rate": 1e-3},
                    "symmetry_cfg": {},
                }
            )
        finally:
            dc.PPO = previous
        self.assertEqual(out["gamma"], 0.99)
        self.assertIsNone(out["rnd_cfg"])
        self.assertIsNone(out["symmetry_cfg"])

    def test_incomplete_symmetry_cfg_becomes_none(self):
        import beamdojo_agents.double_critic as dc

        class FakePPO:
            def __init__(self, policy, symmetry_cfg=None):
                del policy, symmetry_cfg

        previous = dc.PPO
        dc.PPO = FakePPO
        try:
            out = dc._ppo_init_kwargs(
                {
                    "symmetry_cfg": {
                        "use_data_augmentation": True,
                        "use_mirror_loss": False,
                    }
                }
            )
        finally:
            dc.PPO = previous
        self.assertIsNone(out["symmetry_cfg"])


if __name__ == "__main__":
    unittest.main()
