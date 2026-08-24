"""Isaac-free tests for double-critic advantage mix and GAE."""

from __future__ import annotations

import unittest

import numpy as np

from beamdojo_mdp.advantage import W1, W2, combine_advantages, gae_advantages, normalize_adv


class AdvantageTests(unittest.TestCase):
    def test_normalize_zero_mean_unit_std(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        n = normalize_adv(x)
        self.assertAlmostEqual(float(n.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(n.std()), 1.0, places=6)

    def test_combine_weights(self):
        a1 = np.ones(8)
        a2 = np.linspace(-1.0, 1.0, 8)
        mixed = combine_advantages(a1, a2, w1=W1, w2=W2)
        # a1 is constant → normalized to 0; mix is 0.25 * n(a2)
        expected = W2 * normalize_adv(a2)
        np.testing.assert_allclose(mixed, expected, atol=1e-6)

    def test_gae_no_bootstrap_terminal(self):
        rewards = np.array([[1.0], [1.0], [1.0]])
        values = np.array([[0.0], [0.0], [0.0]])
        dones = np.array([[0.0], [0.0], [1.0]])
        last = np.array([[99.0]])  # must not leak in because last step is terminal
        adv, ret = gae_advantages(rewards, values, dones, last, gamma=0.99, lam=0.95)
        self.assertEqual(adv.shape, rewards.shape)
        self.assertLess(float(np.max(np.abs(ret))), 10.0)

    def test_gae_shape_matches(self):
        t, n = 5, 3
        rewards = np.random.randn(t, n)
        values = np.random.randn(t, n)
        dones = np.zeros((t, n))
        last = np.random.randn(n)
        adv, ret = gae_advantages(rewards, values, dones, last)
        self.assertEqual(adv.shape, (t, n))
        self.assertEqual(ret.shape, (t, n))


if __name__ == "__main__":
    unittest.main()
