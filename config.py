"""
Central configuration for the body language capture pipeline.

All tunable parameters are defined here as dataclasses so they can be
easily overridden from CLI arguments or a config file in the future.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union


# ---------------------------------------------------------------------------
# Capture configuration
# ---------------------------------------------------------------------------

@dataclass
class CaptureConfig:
    """Settings for video source and MediaPipe PoseLandmarker."""

    # Video source: integer for webcam index, string for video file path.
    source: Union[int, str] = 0

    # Path to the PoseLandmarker .task model bundle.
    model_path: str = "models/pose_landmarker_lite.task"

    # Detection confidence threshold [0.0, 1.0].
    min_detection_confidence: float = 0.5

    # Tracking confidence threshold [0.0, 1.0].
    min_tracking_confidence: float = 0.5

    # Number of poses to detect (1 = single presenter).
    num_poses: int = 1

    # Target frame dimensions (width, height). None = use source native size.
    frame_width: int | None = 640
    frame_height: int | None = 480

    # Mirror the webcam feed horizontally (natural for self-viewing).
    mirror: bool = True


# ---------------------------------------------------------------------------
# Recording configuration
# ---------------------------------------------------------------------------

@dataclass
class RecordingConfig:
    """Settings for session recording to .npz files."""

    # Directory where .npz session files are saved.
    output_dir: str = "recordings"

    # Whether recording is enabled.
    enabled: bool = False


# ---------------------------------------------------------------------------
# Visualization configuration
# ---------------------------------------------------------------------------

@dataclass
class VisualizationConfig:
    """Settings for skeleton overlay rendering."""

    # Master toggle for the display window.
    enabled: bool = True

    # Skeleton drawing style.
    landmark_radius: int = 4
    connection_thickness: int = 2

    # Color scheme (BGR format for OpenCV).
    # Distinct colors for different body regions.
    torso_color: tuple[int, int, int] = (230, 216, 173)      # Light steel blue
    left_arm_color: tuple[int, int, int] = (78, 205, 196)    # Teal
    right_arm_color: tuple[int, int, int] = (255, 107, 107)  # Coral
    left_leg_color: tuple[int, int, int] = (100, 200, 100)   # Green
    right_leg_color: tuple[int, int, int] = (200, 100, 200)  # Purple
    landmark_color: tuple[int, int, int] = (255, 255, 255)   # White

    # Overlay elements.
    show_fps: bool = True
    show_recording_indicator: bool = True


# ---------------------------------------------------------------------------
# Classification configuration
# ---------------------------------------------------------------------------

@dataclass
class ClassificationConfig:
    """Settings for ST-GCN gesture classification."""

    # Master toggle for gesture classification.
    enabled: bool = False

    # Directory containing trained .pt model files.
    model_dir: str = "models/gesture_classifiers"

    # Number of frames in the sliding classification window.
    # Clips are adaptively sampled to fit this length.
    window_size: int = 128

    # Run inference every N frames (after buffer is full).
    stride: int = 10

    # Minimum confidence to report a gesture prediction.
    confidence_threshold: float = 0.6


# ---------------------------------------------------------------------------
# Pepper robot joint configuration
# ---------------------------------------------------------------------------

# Pepper's controllable joints with their angle limits in radians.
# Format: joint_name -> (min_angle, max_angle)
PEPPER_JOINTS: dict[str, tuple[float, float]] = {
    "HeadYaw":        (-2.0857,  2.0857),
    "HeadPitch":      (-0.7068,  0.6371),
    "LShoulderPitch": (-2.0857,  2.0857),
    "LShoulderRoll":  ( 0.0087,  1.5620),
    "LElbowYaw":      (-2.0857,  2.0857),
    "LElbowRoll":     (-1.5620, -0.0087),
    "LWristYaw":      (-1.8239,  1.8239),
    "RShoulderPitch": (-2.0857,  2.0857),
    "RShoulderRoll":  (-1.5620, -0.0087),
    "RElbowYaw":      (-2.0857,  2.0857),
    "RElbowRoll":     ( 0.0087,  1.5620),
    "RWristYaw":      (-1.8239,  1.8239),
}

# MediaPipe PoseLandmarker landmark indices (from the 33-landmark model).
# Reference: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
LANDMARK_INDICES: dict[str, int] = {
    "nose":            0,
    "left_eye_inner":  1,
    "left_eye":        2,
    "left_eye_outer":  3,
    "right_eye_inner": 4,
    "right_eye":       5,
    "right_eye_outer": 6,
    "left_ear":        7,
    "right_ear":       8,
    "mouth_left":      9,
    "mouth_right":     10,
    "left_shoulder":   11,
    "right_shoulder":  12,
    "left_elbow":      13,
    "right_elbow":     14,
    "left_wrist":      15,
    "right_wrist":     16,
    "left_pinky":      17,
    "right_pinky":     18,
    "left_index":      19,
    "right_index":     20,
    "left_thumb":      21,
    "right_thumb":     22,
    "left_hip":        23,
    "right_hip":       24,
    "left_knee":       25,
    "right_knee":      26,
    "left_ankle":      27,
    "right_ankle":     28,
    "left_heel":       29,
    "right_heel":      30,
    "left_foot_index": 31,
    "right_foot_index":32,
}
