import unittest
import numpy as np
from models.pepper_joint_mapping import compute_pepper_angles
from config import PEPPER_JOINTS

class TestPepperJointMapping(unittest.TestCase):
    def setUp(self):
        # Create a dummy set of (33, 3) landmarks.
        # Initialize all to zero or some default values.
        self.landmarks = np.zeros((33, 3), dtype=np.float32)
        
        # Populate basic indices to avoid complete zero/degenerate collapses where possible.
        # We need landmarks for: nose, left/right ears, left/right shoulders, left/right elbows,
        # left/right wrists, left/right hips, left/right index fingers.
        
        # Hips (around origin)
        self.landmarks[23] = np.array([-0.2, 0.8, 0.0]) # left_hip
        self.landmarks[24] = np.array([0.2, 0.8, 0.0])  # right_hip
        
        # Shoulders (higher up, slightly offset)
        self.landmarks[11] = np.array([-0.25, 0.3, 0.0]) # left_shoulder
        self.landmarks[12] = np.array([0.25, 0.3, 0.0])  # right_shoulder
        
        # Elbows (pointing out/down)
        self.landmarks[13] = np.array([-0.45, 0.5, 0.0]) # left_elbow
        self.landmarks[14] = np.array([0.45, 0.5, 0.0])  # right_elbow
        
        # Wrists (pointing down)
        self.landmarks[15] = np.array([-0.45, 0.7, 0.0]) # left_wrist
        self.landmarks[16] = np.array([0.45, 0.7, 0.0])  # right_wrist
        
        # Index fingers
        self.landmarks[19] = np.array([-0.45, 0.8, 0.0]) # left_index
        self.landmarks[20] = np.array([0.45, 0.8, 0.0])  # right_index
        
        # Nose and ears (head)
        self.landmarks[0] = np.array([0.0, 0.0, 0.1])    # nose
        self.landmarks[7] = np.array([-0.1, -0.05, 0.0]) # left_ear
        self.landmarks[8] = np.array([0.1, -0.05, 0.0])  # right_ear

    def test_compute_pepper_angles_shape_validation(self):
        # Must raise ValueError for incorrect shape
        bad_landmarks = np.zeros((30, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            compute_pepper_angles(bad_landmarks)

        bad_landmarks = np.zeros((33, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            compute_pepper_angles(bad_landmarks)

    def test_compute_pepper_angles_keys(self):
        # Output dictionary should contain exactly the keys in PEPPER_JOINTS
        angles = compute_pepper_angles(self.landmarks)
        self.assertEqual(set(angles.keys()), set(PEPPER_JOINTS.keys()))

    def test_compute_pepper_angles_limits(self):
        # Run calculation on standard dummy landmarks
        angles = compute_pepper_angles(self.landmarks)
        
        for joint, angle in angles.items():
            min_limit, max_limit = PEPPER_JOINTS[joint]
            self.assertTrue(
                min_limit <= angle <= max_limit,
                f"Joint {joint} angle {angle} was not clamped within limits ({min_limit}, {max_limit})"
            )

    def test_extreme_landmarks_are_clamped(self):
        # Set extreme landmarks to force extreme mathematical angles
        # e.g., raising arms completely straight up or way back
        self.landmarks[13] = np.array([-10.0, -10.0, 0.0]) # left elbow far away
        
        angles = compute_pepper_angles(self.landmarks)
        
        # All joints must still be safely within Pepper physical limits
        for joint, angle in angles.items():
            min_limit, max_limit = PEPPER_JOINTS[joint]
            self.assertTrue(
                min_limit <= angle <= max_limit,
                f"Joint {joint} angle {angle} exceeded limits when landmarks were extreme"
            )

if __name__ == "__main__":
    unittest.main()
