"""
Session recorder — serializes pose data to NumPy .npz files.

Accumulates FrameLandmarks during a capture session and writes them
to a self-documenting .npz file on stop. The .npz format is the
standard for Python-based pose estimation pipelines, offering compact
binary storage with fast load times.

File contents:
  - timestamps:       (N,)      int64   — frame timestamps in milliseconds
  - landmarks:        (N,33,4)  float32 — normalized x, y, z, visibility
  - world_landmarks:  (N,33,3)  float32 — real-world coordinates in meters
  - metadata:         ()        str     — JSON-encoded session info

Frames where no pose was detected store NaN values to preserve
temporal alignment (important for replay and Pepper synchronization).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from models.landmark_data import FrameLandmarks

logger = logging.getLogger(__name__)


class SessionRecorder:
    """Records pose landmark sessions to .npz files.

    Usage:
        recorder = SessionRecorder("recordings")
        recorder.start_session(source="webcam", model="lite")
        for frame in frames:
            recorder.record_frame(frame)
        path = recorder.stop_session()
    """

    def __init__(self, output_dir: str) -> None:
        """Initialize the recorder.

        Args:
            output_dir: Directory where .npz files will be saved.
                        Created automatically if it doesn't exist.
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Per-session state.
        self._is_recording = False
        self._frames: list[FrameLandmarks] = []
        self._gesture_results: list[dict | None] = []
        self._session_start: datetime | None = None
        self._session_metadata: dict = {}

    @property
    def is_recording(self) -> bool:
        """True if a session is currently being recorded."""
        return self._is_recording

    @property
    def frame_count(self) -> int:
        """Number of frames recorded in the current session."""
        return len(self._frames)

    def start_session(
        self,
        source: str = "unknown",
        model: str = "unknown",
        fps: float = 30.0,
    ) -> None:
        """Begin a new recording session.

        Args:
            source: Description of the video source (e.g. "webcam", "video.mp4").
            model:  Name of the PoseLandmarker model used.
            fps:    Source frame rate.

        Raises:
            RuntimeError: If a session is already in progress.
        """
        if self._is_recording:
            raise RuntimeError(
                "Cannot start a new session while one is already in progress. "
                "Call stop_session() first."
            )

        self._is_recording = True
        self._frames = []
        self._session_start = datetime.now()
        self._session_metadata = {
            "source": str(source),
            "model": model,
            "fps": fps,
            "start_time": self._session_start.isoformat(),
        }

        logger.info("Recording session started (source=%s, model=%s)", source, model)

    def record_frame(
        self,
        frame_data: FrameLandmarks,
        gesture_results: dict | None = None,
    ) -> None:
        """Record a single frame's landmarks and optional gesture results.

        Args:
            frame_data:      Pose landmarks for the current frame.
                             May have None landmarks if no pose was detected.
            gesture_results: Optional dict of gesture classification results
                             from GestureClassifier.update().

        Raises:
            RuntimeError: If no session is in progress.
        """
        if not self._is_recording:
            raise RuntimeError("No recording session in progress.")

        self._frames.append(frame_data)
        self._gesture_results.append(gesture_results)

    def stop_session(self) -> str:
        """End the current session and write data to a .npz file.

        Returns:
            Absolute path to the saved .npz file.

        Raises:
            RuntimeError: If no session is in progress.
        """
        if not self._is_recording:
            raise RuntimeError("No recording session in progress.")

        self._is_recording = False

        # Compute session statistics.
        num_frames = len(self._frames)
        detected_frames = sum(1 for f in self._frames if f.is_detected)
        duration_sec = 0.0
        if num_frames > 1:
            duration_sec = (
                self._frames[-1].timestamp_ms - self._frames[0].timestamp_ms
            ) / 1000.0

        # Finalize metadata.
        self._session_metadata.update({
            "end_time": datetime.now().isoformat(),
            "frame_count": num_frames,
            "detected_frames": detected_frames,
            "duration_seconds": round(duration_sec, 2),
        })

        # Convert accumulated frames to NumPy arrays.
        timestamps, landmarks, world_landmarks = self._frames_to_arrays()

        # Generate output filename.
        timestamp_str = self._session_start.strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp_str}.npz"
        filepath = self._output_dir / filename

        # Save to .npz with named arrays.
        save_data = {
            "timestamps": timestamps,
            "landmarks": landmarks,
            "world_landmarks": world_landmarks,
            "metadata": json.dumps(self._session_metadata),
        }

        # Add gesture classification data if any results were recorded.
        gesture_arrays = self._gesture_results_to_arrays()
        if gesture_arrays is not None:
            save_data.update(gesture_arrays)

        np.savez_compressed(filepath, **save_data)

        logger.info(
            "Session saved: %s (%d frames, %d detected, %.1fs)",
            filepath,
            num_frames,
            detected_frames,
            duration_sec,
        )

        # Clear session state.
        self._frames = []
        self._gesture_results = []
        self._session_start = None

        return str(filepath.resolve())

    def _frames_to_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert the list of FrameLandmarks to stacked NumPy arrays.

        Returns:
            Tuple of (timestamps, landmarks, world_landmarks):
              - timestamps:      shape (N,)     int64
              - landmarks:       shape (N,33,4) float32
              - world_landmarks: shape (N,33,3) float32
        """
        frame_arrays = [f.to_arrays() for f in self._frames]

        timestamps = np.array(
            [a["timestamp"] for a in frame_arrays], dtype=np.int64
        )
        landmarks = np.stack(
            [a["landmarks"] for a in frame_arrays], axis=0
        )
        world_landmarks = np.stack(
            [a["world_landmarks"] for a in frame_arrays], axis=0
        )

        return timestamps, landmarks, world_landmarks

    def _gesture_results_to_arrays(self) -> dict[str, np.ndarray] | None:
        """Convert gesture classification results to NumPy arrays.

        Returns:
            Dict with 'gesture_labels' and 'gesture_confidences' arrays,
            or None if no gesture data was recorded.
        """
        # Check if any gesture results were actually recorded.
        has_gestures = any(r is not None for r in self._gesture_results)
        if not has_gestures:
            return None

        # Collect all gesture type names from the first non-None result.
        gesture_types: list[str] = []
        for r in self._gesture_results:
            if r is not None:
                gesture_types = sorted(r.keys())
                break

        if not gesture_types:
            return None

        num_frames = len(self._gesture_results)
        num_gestures = len(gesture_types)

        # Build arrays.
        labels = []
        confidences = np.zeros((num_frames, num_gestures), dtype=np.float32)

        for t, result in enumerate(self._gesture_results):
            if result is None:
                labels.append("")
                continue

            # Combine labels for this frame.
            frame_labels = []
            for g_idx, g_type in enumerate(gesture_types):
                if g_type in result:
                    frame_labels.append(result[g_type].label)
                    confidences[t, g_idx] = result[g_type].confidence

            labels.append("|".join(frame_labels))

        return {
            "gesture_labels": np.array(labels, dtype=object),
            "gesture_confidences": confidences,
            "gesture_types": np.array(gesture_types, dtype=object),
        }
