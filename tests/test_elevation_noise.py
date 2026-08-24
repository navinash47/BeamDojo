"""Isaac-free tests for paper Table IX elevation-map noise."""

from __future__ import annotations

import unittest

import numpy as np

from beamdojo_mdp.elevation_noise import dilate_on_cells, maybe_repeat_map, tilt_height_grid
from beamdojo_mdp.heightfield import BEAM_OFF_Z, BEAM_ON_Z


class ElevationNoiseTests(unittest.TestCase):
    def test_tilt_raises_one_edge(self):
        n = 3
        hz = np.zeros((1, n * n))
        out = tilt_height_grid(hz, n, hx=np.array([0.03]), hy=np.array([0.0]))
        self.assertAlmostEqual(float(out[0, 0]), -0.03, places=6)
        self.assertAlmostEqual(float(out[0, -n]), 0.03, places=6)

    def test_dilate_spreads_on_cells(self):
        n = 3
        hz = np.full((1, n * n), BEAM_OFF_Z)
        hz[0, 4] = BEAM_ON_Z  # center
        out = dilate_on_cells(hz, n, extend=np.array([True]))
        # center plus 4-neighbors become on
        self.assertGreater(int((out[0] == BEAM_ON_Z).sum()), 1)
        skipped = dilate_on_cells(hz, n, extend=np.array([False]))
        np.testing.assert_allclose(skipped, hz)

    def test_map_repeat_keeps_previous(self):
        cur = np.array([[1.0, 2.0]])
        prev = np.array([[9.0, 8.0]])
        out = maybe_repeat_map(cur, prev, repeat_mask=np.array([True]))
        np.testing.assert_allclose(out, prev)
        out2 = maybe_repeat_map(cur, prev, repeat_mask=np.array([False]))
        np.testing.assert_allclose(out2, cur)


if __name__ == "__main__":
    unittest.main()
