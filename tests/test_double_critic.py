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


if __name__ == "__main__":
    unittest.main()
