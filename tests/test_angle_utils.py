import unittest
import numpy as np
from utils.angle_utils import (
    vector_between,
    calculate_angle,
    rotation_angle_around_axis,
    clamp_angle,
)

class TestAngleUtils(unittest.TestCase):
    def test_vector_between(self):
        # Simple displacement vector test
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 6.0, 8.0])
        expected = np.array([3.0, 4.0, 5.0])
        np.testing.assert_allclose(vector_between(a, b), expected)

    def test_calculate_angle_right_angle(self):
        # 90 degrees (pi/2) angle test
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        angle = calculate_angle(a, b, c)
        self.assertAlmostEqual(angle, np.pi / 2.0)

    def test_calculate_angle_straight_line(self):
        # 180 degrees (pi) angle test
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([-1.0, 0.0, 0.0])
        angle = calculate_angle(a, b, c)
        self.assertAlmostEqual(angle, np.pi)

    def test_calculate_angle_coincident_points(self):
        # Degenerate case test: vertex and endpoint are coincident
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        angle = calculate_angle(a, b, c)
        self.assertEqual(angle, 0.0)

    def test_rotation_angle_around_axis_simple(self):
        # 90 degree rotation in XY plane around Z axis
        vec_from = np.array([1.0, 0.0, 0.0])
        vec_to = np.array([0.0, 1.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        angle = rotation_angle_around_axis(vec_from, vec_to, axis)
        self.assertAlmostEqual(angle, np.pi / 2.0)

    def test_rotation_angle_around_axis_negative(self):
        # -90 degree rotation in XY plane around Z axis
        vec_from = np.array([1.0, 0.0, 0.0])
        vec_to = np.array([0.0, -1.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        angle = rotation_angle_around_axis(vec_from, vec_to, axis)
        self.assertAlmostEqual(angle, -np.pi / 2.0)

    def test_rotation_angle_around_axis_degenerate_axis(self):
        # Zero axis should return 0.0
        vec_from = np.array([1.0, 0.0, 0.0])
        vec_to = np.array([0.0, 1.0, 0.0])
        axis = np.array([0.0, 0.0, 0.0])
        angle = rotation_angle_around_axis(vec_from, vec_to, axis)
        self.assertEqual(angle, 0.0)

    def test_clamp_angle(self):
        # Test clamping values within, below, and above limits
        self.assertEqual(clamp_angle(0.5, 0.0, 1.0), 0.5)
        self.assertEqual(clamp_angle(-0.5, 0.0, 1.0), 0.0)
        self.assertEqual(clamp_angle(1.5, 0.0, 1.0), 1.0)

if __name__ == "__main__":
    unittest.main()
