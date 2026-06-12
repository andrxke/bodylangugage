"""
Skeleton overlay renderer for video frames.

Draws the 33-point pose skeleton on video frames with color-coded
body regions, confidence-based opacity, and HUD overlays (FPS counter,
recording indicator). Uses MediaPipe's POSE_CONNECTIONS for edge
definitions but applies custom styling for clarity.

Body regions and their colors are configurable via VisualizationConfig.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from config import VisualizationConfig
from models.landmark_data import FrameLandmarks

if TYPE_CHECKING:
    from classification.gesture_classifier import GestureResult


# ---------------------------------------------------------------------------
# MediaPipe pose connection definitions, grouped by body region.
# Each tuple is (start_landmark_index, end_landmark_index).
# ---------------------------------------------------------------------------

# Torso connections (shoulders, hips, and the lines between them).
_TORSO_CONNECTIONS = [
    (11, 12),  # Left shoulder → Right shoulder
    (11, 23),  # Left shoulder → Left hip
    (12, 24),  # Right shoulder → Right hip
    (23, 24),  # Left hip → Right hip
]

# Left arm connections.
_LEFT_ARM_CONNECTIONS = [
    (11, 13),  # Left shoulder → Left elbow
    (13, 15),  # Left elbow → Left wrist
    (15, 17),  # Left wrist → Left pinky
    (15, 19),  # Left wrist → Left index
    (15, 21),  # Left wrist → Left thumb
    (17, 19),  # Left pinky → Left index (palm line)
]

# Right arm connections.
_RIGHT_ARM_CONNECTIONS = [
    (12, 14),  # Right shoulder → Right elbow
    (14, 16),  # Right elbow → Right wrist
    (16, 18),  # Right wrist → Right pinky
    (16, 20),  # Right wrist → Right index
    (16, 22),  # Right wrist → Right thumb
    (18, 20),  # Right pinky → Right index (palm line)
]

# Left leg connections.
_LEFT_LEG_CONNECTIONS = [
    (23, 25),  # Left hip → Left knee
    (25, 27),  # Left knee → Left ankle
    (27, 29),  # Left ankle → Left heel
    (27, 31),  # Left ankle → Left foot index
    (29, 31),  # Left heel → Left foot index
]

# Right leg connections.
_RIGHT_LEG_CONNECTIONS = [
    (24, 26),  # Right hip → Right knee
    (26, 28),  # Right knee → Right ankle
    (28, 30),  # Right ankle → Right heel
    (28, 32),  # Right ankle → Right foot index
    (30, 32),  # Right heel → Right foot index
]

# Head/face connections (simplified — just the key structural lines).
_HEAD_CONNECTIONS = [
    (0, 1),    # Nose → Left eye inner
    (0, 4),    # Nose → Right eye inner
    (1, 2),    # Left eye inner → Left eye
    (2, 3),    # Left eye → Left eye outer
    (3, 7),    # Left eye outer → Left ear
    (4, 5),    # Right eye inner → Right eye
    (5, 6),    # Right eye → Right eye outer
    (6, 8),    # Right eye outer → Right ear
    (9, 10),   # Mouth left → Mouth right
]


class SkeletonRenderer:
    """Renders pose skeleton overlays on video frames.

    Usage:
        renderer = SkeletonRenderer(config)
        annotated = renderer.draw(frame, frame_landmarks, is_recording=True)
        cv2.imshow("Pose", annotated)

    The renderer maintains an internal FPS counter that updates
    based on the time between draw() calls.
    """

    def __init__(self, config: VisualizationConfig) -> None:
        """Initialize the renderer with display configuration.

        Args:
            config: Visualization settings (colors, sizes, toggles).
        """
        self._config = config

        # FPS tracking.
        self._prev_time = time.time()
        self._fps = 0.0

        # Build the connection-to-color mapping.
        self._connection_colors = self._build_color_map()

    def _build_color_map(self) -> list[tuple[tuple[int, int], tuple[int, ...]]]:
        """Create a list of (connection, color) pairs for drawing.

        Returns:
            List of ((start_idx, end_idx), BGR_color) tuples.
        """
        cfg = self._config
        color_map = []

        for conn in _TORSO_CONNECTIONS:
            color_map.append((conn, cfg.torso_color))
        for conn in _LEFT_ARM_CONNECTIONS:
            color_map.append((conn, cfg.left_arm_color))
        for conn in _RIGHT_ARM_CONNECTIONS:
            color_map.append((conn, cfg.right_arm_color))
        for conn in _LEFT_LEG_CONNECTIONS:
            color_map.append((conn, cfg.left_leg_color))
        for conn in _RIGHT_LEG_CONNECTIONS:
            color_map.append((conn, cfg.right_leg_color))
        for conn in _HEAD_CONNECTIONS:
            color_map.append((conn, cfg.torso_color))

        return color_map

    def draw(
        self,
        frame: np.ndarray,
        frame_landmarks: FrameLandmarks | None,
        is_recording: bool = False,
        gesture_results: dict[str, GestureResult] | None = None,
    ) -> np.ndarray:
        """Draw the pose skeleton and HUD elements on a frame.

        Args:
            frame:           BGR frame to annotate (will be modified in-place).
            frame_landmarks: Detected pose landmarks, or None if no detection.
            is_recording:    If True, shows a red recording indicator.
            gesture_results: Optional dict of gesture classification results.

        Returns:
            The annotated BGR frame.
        """
        # Update FPS counter.
        self._update_fps()

        # Draw the skeleton if landmarks are available.
        if frame_landmarks is not None and frame_landmarks.is_detected:
            self._draw_skeleton(frame, frame_landmarks)

        # Draw HUD overlays.
        if self._config.show_fps:
            self._draw_fps(frame)
        if self._config.show_recording_indicator and is_recording:
            self._draw_recording_indicator(frame)
        if gesture_results:
            self._draw_gesture_labels(frame, gesture_results)

        return frame

    def _draw_skeleton(
        self,
        frame: np.ndarray,
        frame_landmarks: FrameLandmarks,
    ) -> None:
        """Draw skeleton connections and landmark points on the frame.

        Args:
            frame:           BGR frame to draw on.
            frame_landmarks: Detected pose landmarks (must have is_detected=True).
        """
        h, w = frame.shape[:2]
        landmarks = frame_landmarks.landmarks  # (33, 4): x, y, z, vis

        # Pre-compute pixel coordinates for all landmarks.
        # x and y are normalized [0, 1], need to scale to frame dimensions.
        pixel_coords = np.zeros((33, 2), dtype=np.int32)
        pixel_coords[:, 0] = (landmarks[:, 0] * w).astype(np.int32)
        pixel_coords[:, 1] = (landmarks[:, 1] * h).astype(np.int32)
        visibilities = landmarks[:, 3]

        # Draw connections (lines between landmarks).
        for (start_idx, end_idx), color in self._connection_colors:
            # Only draw if both landmarks are reasonably visible.
            if visibilities[start_idx] < 0.5 or visibilities[end_idx] < 0.5:
                continue

            pt1 = tuple(pixel_coords[start_idx])
            pt2 = tuple(pixel_coords[end_idx])

            cv2.line(
                frame,
                pt1,
                pt2,
                color,
                self._config.connection_thickness,
                lineType=cv2.LINE_AA,
            )

        # Draw landmark points (circles).
        for i in range(33):
            if visibilities[i] < 0.3:
                continue

            # Scale circle opacity with visibility confidence.
            alpha = min(1.0, visibilities[i])
            color = tuple(int(c * alpha) for c in self._config.landmark_color)

            cv2.circle(
                frame,
                tuple(pixel_coords[i]),
                self._config.landmark_radius,
                color,
                -1,  # Filled circle.
                lineType=cv2.LINE_AA,
            )

    def _update_fps(self) -> None:
        """Update the rolling FPS estimate."""
        now = time.time()
        elapsed = now - self._prev_time
        if elapsed > 0:
            # Exponential moving average for smooth FPS display.
            instant_fps = 1.0 / elapsed
            self._fps = 0.9 * self._fps + 0.1 * instant_fps
        self._prev_time = now

    def _draw_fps(self, frame: np.ndarray) -> None:
        """Draw the FPS counter in the top-left corner.

        Args:
            frame: BGR frame to draw on.
        """
        text = f"FPS: {self._fps:.1f}"
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),  # Green text.
            2,
            cv2.LINE_AA,
        )

    def _draw_recording_indicator(self, frame: np.ndarray) -> None:
        """Draw a red recording dot in the top-right corner.

        Args:
            frame: BGR frame to draw on.
        """
        h, w = frame.shape[:2]

        # Red filled circle.
        center = (w - 30, 30)
        cv2.circle(frame, center, 10, (0, 0, 255), -1, cv2.LINE_AA)

        # "REC" label.
        cv2.putText(
            frame,
            "REC",
            (w - 80, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_gesture_labels(
        self,
        frame: np.ndarray,
        gesture_results: dict[str, GestureResult],
    ) -> None:
        """Draw gesture classification labels below the FPS counter.

        Shows each active gesture with a checkmark/cross and confidence
        percentage. Green for positive states, red for negative.

        Args:
            frame:           BGR frame to draw on.
            gesture_results: Dict of gesture type → GestureResult.
        """
        y_offset = 60  # Start below the FPS counter.

        for gesture_type, result in gesture_results.items():
            if result.is_positive:
                symbol = "\u2713"  # ✓
                color = (0, 200, 0)  # Green (BGR).
            else:
                symbol = "\u2717"  # ✗
                color = (0, 0, 200)  # Red (BGR).

            label_text = gesture_type.upper().replace("_", " ")
            confidence_pct = f"{result.confidence:.0%}"
            text = f"{label_text} {symbol} {confidence_pct}"

            cv2.putText(
                frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

            y_offset += 25
