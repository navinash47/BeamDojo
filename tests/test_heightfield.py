"""Unit tests for dual-terrain heightfield math (no Isaac / CUDA required)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from beamdojo_mdp.heightfield import (
    BEAM_OFF_Z,
    BEAM_ON_Z,
    beam_surface_z,
    foothold_off_count,
    stone_surface_z,
    yaw_from_quat_wxyz,
    yaw_grid_xy,
)


class BeamHeightfieldTests(unittest.TestCase):
    def test_on_beam_center(self):
        z = beam_surface_z(
            np.array([4.0]),
            np.array([0.0]),
            origin_x=np.array([0.0]),
            origin_y=np.array([0.0]),
            length=8.0,
            width=0.20,
        )
        np.testing.assert_allclose(z, [BEAM_ON_Z])

    def test_off_beam_laterally(self):
        z = beam_surface_z(
            np.array([4.0]),
            np.array([0.30]),
            origin_x=np.array([0.0]),
            origin_y=np.array([0.0]),
            length=8.0,
            width=0.20,
        )
        np.testing.assert_allclose(z, [BEAM_OFF_Z])

    def test_per_env_width(self):
        x = np.array([1.0, 1.0])
        y = np.array([0.12, 0.12])
        origin = np.array([0.0, 0.0])
        z = beam_surface_z(x, y, origin, origin, 8.0, width=np.array([0.40, 0.20]))
        np.testing.assert_allclose(z, [BEAM_ON_Z, BEAM_OFF_Z])

    def test_env_origin_shift(self):
        z = beam_surface_z(
            np.array([10.0]),
            np.array([3.0]),
            origin_x=np.array([6.0]),
            origin_y=np.array([3.0]),
            length=8.0,
            width=0.20,
        )
        np.testing.assert_allclose(z, [BEAM_ON_Z])

    def test_foothold_off_count_all_off(self):
        sx = np.array([[1.0, 1.02, 0.98]])
        sy = np.array([[0.4, 0.41, 0.39]])
        n = foothold_off_count(
            sx,
            sy,
            origin_x=np.array([0.0]),
            origin_y=np.array([0.0]),
            length=8.0,
            width=0.20,
        )
        np.testing.assert_allclose(n, [3.0])

    def test_foothold_on_beam_zero(self):
        sx = np.array([[2.0, 2.02, 1.98]])
        sy = np.array([[0.0, 0.02, -0.02]])
        n = foothold_off_count(
            sx,
            sy,
            origin_x=np.array([0.0]),
            origin_y=np.array([0.0]),
            length=8.0,
            width=0.20,
        )
        np.testing.assert_allclose(n, [0.0])

    def test_stones_on_pad_and_gap(self):
        origin = np.array([0.0])
        on = stone_surface_z(
            np.array([0.05]),
            np.array([0.0]),
            origin,
            origin,
            length=8.0,
            stone_size=0.20,
            gap=0.10,
        )
        off = stone_surface_z(
            np.array([0.25]),
            np.array([0.0]),
            origin,
            origin,
            length=8.0,
            stone_size=0.20,
            gap=0.10,
        )
        np.testing.assert_allclose(on, [BEAM_ON_Z])
        np.testing.assert_allclose(off, [BEAM_OFF_Z])

    def test_yaw_grid_identity(self):
        px = np.array([0.0])
        py = np.array([0.0])
        yaw = np.array([0.0])
        gx, gy = yaw_grid_xy(px, py, yaw, n=3, extent=2.0)
        self.assertEqual(gx.shape, (1, 9))
        np.testing.assert_allclose(gx[0, 0], -1.0, atol=1e-6)
        np.testing.assert_allclose(gy[0, 0], -1.0, atol=1e-6)

    def test_yaw_from_identity_quat(self):
        q = np.array([[1.0, 0.0, 0.0, 0.0]])
        yaw = yaw_from_quat_wxyz(q)
        np.testing.assert_allclose(yaw, [0.0], atol=1e-6)

    def test_yaw_from_z_rotation(self):
        a = math.pi / 2
        q = np.array([[math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]])
        yaw = yaw_from_quat_wxyz(q)
        np.testing.assert_allclose(yaw, [a], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
