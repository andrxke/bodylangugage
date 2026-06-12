"""
Data classes for pose landmark storage.

Defines the core data structure used throughout the pipeline to represent
a single frame's pose detection results. Designed for:
  - Efficient NumPy-based storage and batch operations
  - Direct compatibility with .npz serialization
  - Downstream use by Pepper joint mapping and body language analysis

Coordinate systems:
  - `landmarks`:       Normalized (x, y in [0,1], z relative to hips).
                        Used for drawing overlays on the video frame.
  - `world_landmarks`: Real-world coordinates in meters, with the hip
                        midpoint as origin. Used for 3D angle calculations
                        and Pepper joint mapping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Number of landmarks produced by the MediaPipe PoseLandmarker model.
NUM_POSE_LANDMARKS = 33

# Columns per landmark in the normalized array: x, y, z, visibility.
LANDMARK_COLS = 4

# Columns per landmark in the world array: x, y, z (meters).
WORLD_LANDMARK_COLS = 3


@dataclass
class FrameLandmarks:
    """Pose landmarks for a single video frame.

    Attributes:
        timestamp_ms: Frame timestamp in milliseconds (monotonically
                      increasing within a session).
        landmarks:    Normalized landmark coordinates, shape (33, 4).
                      Columns: [x, y, z, visibility].
                      None if no pose was detected in this frame.
        world_landmarks: Real-world 3D coordinates in meters, shape (33, 3).
                         Columns: [x, y, z].
                         None if no pose was detected in this frame.
    """

    timestamp_ms: int
    landmarks: np.ndarray | None        # (33, 4): x, y, z, visibility
    world_landmarks: np.ndarray | None  # (33, 3): x, y, z in meters

    @property
    def is_detected(self) -> bool:
        """True if a pose was successfully detected in this frame."""
        return self.landmarks is not None

    def get_landmark(self, index: int) -> np.ndarray | None:
        """Get a single normalized landmark by index.

        Args:
            index: Landmark index (0–32). Use LANDMARK_INDICES from config.

        Returns:
            Array of [x, y, z, visibility], or None if no pose detected.
        """
        if self.landmarks is None:
            return None
        return self.landmarks[index]

    def get_world_landmark(self, index: int) -> np.ndarray | None:
        """Get a single world-coordinate landmark by index.

        Args:
            index: Landmark index (0–32).

        Returns:
            Array of [x, y, z] in meters, or None if no pose detected.
        """
        if self.world_landmarks is None:
            return None
        return self.world_landmarks[index]

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Convert to dict of arrays for .npz storage.

        If no pose was detected, landmark arrays are filled with NaN
        to preserve temporal alignment in the recording.

        Returns:
            Dictionary with 'timestamp', 'landmarks', 'world_landmarks' keys.
        """
        if self.landmarks is not None:
            lm = self.landmarks.astype(np.float32)
        else:
            # NaN placeholder preserves array shape for frames without detection.
            lm = np.full(
                (NUM_POSE_LANDMARKS, LANDMARK_COLS), np.nan, dtype=np.float32
            )

        if self.world_landmarks is not None:
            wlm = self.world_landmarks.astype(np.float32)
        else:
            wlm = np.full(
                (NUM_POSE_LANDMARKS, WORLD_LANDMARK_COLS),
                np.nan,
                dtype=np.float32,
            )

        return {
            "timestamp": np.array(self.timestamp_ms, dtype=np.int64),
            "landmarks": lm,
            "world_landmarks": wlm,
        }

    @classmethod
    def from_arrays(
        cls,
        timestamp: int,
        landmarks: np.ndarray,
        world_landmarks: np.ndarray,
    ) -> FrameLandmarks:
        """Reconstruct a FrameLandmarks from stored arrays.

        NaN arrays (from frames where no pose was detected) are converted
        back to None.

        Args:
            timestamp:       Timestamp in milliseconds.
            landmarks:       Shape (33, 4) float array.
            world_landmarks: Shape (33, 3) float array.

        Returns:
            Reconstructed FrameLandmarks instance.
        """
        # If the entire array is NaN, this frame had no detection.
        lm = None if np.all(np.isnan(landmarks)) else landmarks
        wlm = None if np.all(np.isnan(world_landmarks)) else world_landmarks

        return cls(
            timestamp_ms=int(timestamp),
            landmarks=lm,
            world_landmarks=wlm,
        )

    @classmethod
    def from_mediapipe_result(
        cls,
        result,
        timestamp_ms: int,
    ) -> FrameLandmarks:
        """Construct from a MediaPipe PoseLandmarkerResult.

        Extracts the first detected pose (single-presenter assumption)
        and converts the protobuf landmark objects to NumPy arrays.

        Args:
            result:       PoseLandmarkerResult from MediaPipe.
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            FrameLandmarks with extracted data, or with None landmarks
            if no pose was detected.
        """
        if not result.pose_landmarks:
            return cls(
                timestamp_ms=timestamp_ms,
                landmarks=None,
                world_landmarks=None,
            )

        # Extract the first (and typically only) detected pose.
        pose = result.pose_landmarks[0]
        world_pose = result.pose_world_landmarks[0]

        # Convert normalized landmarks to (33, 4) array.
        landmarks = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in pose],
            dtype=np.float32,
        )

        # Convert world landmarks to (33, 3) array.
        world_landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in world_pose],
            dtype=np.float32,
        )

        return cls(
            timestamp_ms=timestamp_ms,
            landmarks=landmarks,
            world_landmarks=world_landmarks,
        )
