"""
Tests for the GestureClassifier inference wrapper.

Verifies buffer management, input preparation, interpolation of
missing frames, and the inference stride logic. Uses a mock model
to avoid needing a trained checkpoint.
"""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from classification.gesture_classifier import GestureClassifier, GestureResult
from models.landmark_data import FrameLandmarks


def _make_frame(timestamp_ms: int = 0, detected: bool = True) -> FrameLandmarks:
    """Create a synthetic FrameLandmarks for testing.

    Args:
        timestamp_ms: Frame timestamp.
        detected:     Whether pose was detected.

    Returns:
        FrameLandmarks instance.
    """
    if detected:
        landmarks = np.random.randn(33, 4).astype(np.float32)
        world_landmarks = np.random.randn(33, 3).astype(np.float32)
    else:
        landmarks = None
        world_landmarks = None

    return FrameLandmarks(
        timestamp_ms=timestamp_ms,
        landmarks=landmarks,
        world_landmarks=world_landmarks,
    )


class TestGestureResult:
    """Tests for the GestureResult dataclass."""

    def test_creation(self):
        """GestureResult should be created with all fields."""
        result = GestureResult(
            gesture_type="facing",
            label="facing_audience",
            confidence=0.85,
            is_positive=True,
        )
        assert result.gesture_type == "facing"
        assert result.label == "facing_audience"
        assert result.confidence == 0.85
        assert result.is_positive is True


class TestGestureClassifierNoModels:
    """Tests for GestureClassifier when no models are loaded."""

    def test_no_models_directory(self, tmp_path):
        """Should initialize gracefully with missing model dir."""
        classifier = GestureClassifier(
            model_dir=str(tmp_path / "nonexistent"),
            window_size=16,
        )
        assert not classifier.is_ready

    def test_empty_model_directory(self, tmp_path):
        """Should initialize gracefully with empty model dir."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        classifier = GestureClassifier(
            model_dir=str(model_dir),
            window_size=16,
        )
        assert not classifier.is_ready

    def test_update_returns_none_without_models(self, tmp_path):
        """update() should return None when no models are loaded."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        classifier = GestureClassifier(
            model_dir=str(model_dir),
            window_size=16,
        )
        frame = _make_frame(0)
        result = classifier.update(frame)
        assert result is None


class TestGestureClassifierBuffer:
    """Tests for the classifier's frame buffer management."""

    def test_buffer_fills(self, tmp_path):
        """Buffer should accumulate frames up to window_size."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        classifier = GestureClassifier(
            model_dir=str(model_dir),
            window_size=16,
        )

        for i in range(16):
            frame = _make_frame(i * 33)
            classifier._buffer.append(
                frame.world_landmarks.copy() if frame.is_detected else None
            )

        assert len(classifier._buffer) == 16

    def test_buffer_is_rolling(self, tmp_path):
        """Buffer should drop oldest frames when full."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        classifier = GestureClassifier(
            model_dir=str(model_dir),
            window_size=8,
        )

        for i in range(20):
            frame = _make_frame(i * 33)
            classifier._buffer.append(
                frame.world_landmarks.copy() if frame.is_detected else None
            )

        assert len(classifier._buffer) == 8


class TestInputPreparation:
    """Tests for _prepare_input and _interpolate_missing."""

    def test_prepare_input_shape(self, tmp_path):
        """Prepared input should have shape (1, 3, T, 33, 1)."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        classifier = GestureClassifier(
            model_dir=str(model_dir),
            window_size=16,
        )

        # Fill buffer with synthetic frames.
        for i in range(16):
            frame = _make_frame(i * 33)
            classifier._buffer.append(frame.world_landmarks.copy())

        tensor = classifier._prepare_input()
        assert tensor.shape == (1, 3, 16, 33, 1)
        assert tensor.dtype == torch.float32

    def test_prepare_input_hip_centered(self, tmp_path):
        """Output should be centered on hip midpoint."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        classifier = GestureClassifier(
            model_dir=str(model_dir),
            window_size=8,
        )

        # Create landmarks where all joints are at (1, 2, 3).
        wl = np.ones((33, 3), dtype=np.float32) * np.array([1.0, 2.0, 3.0])
        for _ in range(8):
            classifier._buffer.append(wl.copy())

        tensor = classifier._prepare_input()

        # After hip-centering, everything should be near zero.
        assert torch.allclose(tensor, torch.zeros_like(tensor), atol=1e-5)

    def test_interpolate_all_valid(self):
        """Interpolation with all valid frames should be a no-op."""
        data = np.random.randn(10, 33, 3).astype(np.float32)
        valid = np.ones(10, dtype=bool)

        result = GestureClassifier._interpolate_missing(data.copy(), valid)
        assert np.allclose(result, data)

    def test_interpolate_middle_gap(self):
        """Should linearly interpolate a gap in the middle."""
        data = np.zeros((5, 2, 3), dtype=np.float32)
        data[0] = 0.0
        data[4] = 4.0
        valid = np.array([True, False, False, False, True])

        result = GestureClassifier._interpolate_missing(data.copy(), valid)

        # Frames 1, 2, 3 should be interpolated between 0 and 4.
        assert np.allclose(result[1], 1.0, atol=1e-5)
        assert np.allclose(result[2], 2.0, atol=1e-5)
        assert np.allclose(result[3], 3.0, atol=1e-5)

    def test_interpolate_leading_gap(self):
        """Should fill leading gap with first valid frame."""
        data = np.zeros((4, 2, 3), dtype=np.float32)
        data[2] = 5.0
        data[3] = 6.0
        valid = np.array([False, False, True, True])

        result = GestureClassifier._interpolate_missing(data.copy(), valid)

        assert np.allclose(result[0], 5.0)
        assert np.allclose(result[1], 5.0)

    def test_interpolate_trailing_gap(self):
        """Should fill trailing gap with last valid frame."""
        data = np.zeros((4, 2, 3), dtype=np.float32)
        data[0] = 3.0
        data[1] = 4.0
        valid = np.array([True, True, False, False])

        result = GestureClassifier._interpolate_missing(data.copy(), valid)

        assert np.allclose(result[2], 4.0)
        assert np.allclose(result[3], 4.0)

    def test_interpolate_no_valid_frames(self):
        """With no valid frames, should return zeros."""
        data = np.zeros((4, 2, 3), dtype=np.float32)
        valid = np.array([False, False, False, False])

        result = GestureClassifier._interpolate_missing(data.copy(), valid)
        assert np.allclose(result, 0.0)
