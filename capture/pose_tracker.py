"""
MediaPipe PoseLandmarker wrapper.

Provides a unified interface for pose detection using the modern
MediaPipe Tasks API. Supports two modes:
  - VIDEO mode:       Synchronous, for pre-recorded video files.
  - LIVE_STREAM mode: Asynchronous with callback, for live webcam feeds.

The mode is automatically selected based on the source type in the config.
Results are converted into the pipeline's FrameLandmarks data structure
for downstream consumption by recording, visualization, and Pepper mapping.
"""

from __future__ import annotations

import threading
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

from config import CaptureConfig
from models.landmark_data import FrameLandmarks


class PoseTracker:
    """Wraps MediaPipe PoseLandmarker for pose skeleton detection.

    Automatically selects LIVE_STREAM mode for webcam input and
    VIDEO mode for pre-recorded files.

    Usage:
        config = CaptureConfig(source=0)  # webcam
        with PoseTracker(config) as tracker:
            result = tracker.process_frame(rgb_frame, timestamp_ms)

    Attributes:
        is_live: True if using LIVE_STREAM mode (webcam).
    """

    def __init__(self, config: CaptureConfig) -> None:
        """Initialize the PoseLandmarker with the given configuration.

        Args:
            config: Capture configuration specifying model path,
                    confidence thresholds, and source type.

        Raises:
            FileNotFoundError: If the model .task file does not exist.
        """
        self._config = config
        self.is_live = isinstance(config.source, int)

        # Verify model file exists.
        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"PoseLandmarker model not found at '{model_path}'. "
                f"Run 'python download_model.py' to download it."
            )

        # Thread-safe storage for the latest async result (LIVE_STREAM mode).
        self._latest_result: FrameLandmarks | None = None
        self._result_lock = threading.Lock()

        # Build the landmarker options based on source type.
        if self.is_live:
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(model_path),
                ),
                running_mode=RunningMode.LIVE_STREAM,
                num_poses=config.num_poses,
                min_pose_detection_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
                result_callback=self._live_stream_callback,
            )
        else:
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(model_path),
                ),
                running_mode=RunningMode.VIDEO,
                num_poses=config.num_poses,
                min_pose_detection_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
            )

        self._landmarker = PoseLandmarker.create_from_options(options)

    def _live_stream_callback(
        self,
        result,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        """Callback for LIVE_STREAM mode — stores the latest result.

        This runs on MediaPipe's internal thread, so we use a lock to
        ensure thread safety when the main loop reads the result.

        Args:
            result:       PoseLandmarkerResult from MediaPipe.
            output_image: The input image (unused).
            timestamp_ms: Timestamp of the processed frame.
        """
        frame_data = FrameLandmarks.from_mediapipe_result(result, timestamp_ms)
        with self._result_lock:
            self._latest_result = frame_data

    def process_frame(
        self,
        rgb_frame: np.ndarray,
        timestamp_ms: int,
    ) -> FrameLandmarks | None:
        """Process a single frame and return pose landmarks.

        Behavior depends on mode:
          - VIDEO:       Runs detection synchronously and returns the result.
          - LIVE_STREAM: Sends the frame for async processing and returns
                         the latest available result (may be from a previous
                         frame). Returns None if no result is available yet.

        Args:
            rgb_frame:    Frame in RGB format (H, W, 3).
            timestamp_ms: Monotonically increasing timestamp in milliseconds.

        Returns:
            FrameLandmarks with detection results, or None if no result
            is available yet (LIVE_STREAM only).
        """
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        if self.is_live:
            # Asynchronous — send frame and return latest available result.
            self._landmarker.detect_async(mp_image, timestamp_ms)
            return self.get_latest_result()
        else:
            # Synchronous — block until detection completes.
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
            return FrameLandmarks.from_mediapipe_result(result, timestamp_ms)

    def get_latest_result(self) -> FrameLandmarks | None:
        """Retrieve the latest detection result (thread-safe).

        Used primarily in LIVE_STREAM mode where results arrive
        asynchronously via callback.

        Returns:
            The most recent FrameLandmarks, or None if no result yet.
        """
        with self._result_lock:
            return self._latest_result

    def close(self) -> None:
        """Release the PoseLandmarker resources."""
        self._landmarker.close()

    # -- Context manager support --

    def __enter__(self) -> PoseTracker:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
