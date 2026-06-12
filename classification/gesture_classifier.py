"""
Real-time gesture classifier using trained ST-GCN models.

Maintains a rolling buffer of recent pose landmarks and runs
ST-GCN inference at regular intervals using a sliding window.
Designed to integrate seamlessly into the existing capture loop
in main.py.

Usage:
    classifier = GestureClassifier("models/gesture_classifiers")

    # In the frame loop:
    results = classifier.update(frame_data)
    if results:
        for name, result in results.items():
            print(f"{name}: {result.label} ({result.confidence:.0%})")
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from classification.gesture_labels import GESTURE_TYPES
from classification.graph import MediaPipeGraph
from classification.stgcn import STGCN
from models.landmark_data import FrameLandmarks

logger = logging.getLogger(__name__)


@dataclass
class GestureResult:
    """Classification result for a single gesture type.

    Attributes:
        gesture_type: Gesture type key (e.g. "facing").
        label:        Predicted label string (e.g. "facing_audience").
        confidence:   Prediction confidence (0.0 – 1.0).
        is_positive:  True if the positive class was predicted.
    """

    gesture_type: str
    label: str
    confidence: float
    is_positive: bool


class GestureClassifier:
    """Sliding-window gesture classifier using trained ST-GCN models.

    Accumulates pose landmarks in a rolling buffer. Once the buffer
    is full, classifies the gesture state every `stride` frames.

    Attributes:
        window_size: Number of frames in the classification window.
        stride:      Classify every N frames.
        models:      Dict of loaded ST-GCN models keyed by gesture type.
    """

    def __init__(
        self,
        model_dir: str,
        window_size: int = 128,
        stride: int = 10,
    ) -> None:
        """Initialize the classifier.

        Args:
            model_dir:   Directory containing trained .pt model files
                         and their _config.json companions.
            window_size: Number of frames in the sliding window.
            stride:      Run inference every N frames after the buffer
                         is full.
        """
        self.window_size = window_size
        self.stride = stride

        # Rolling buffer of world_landmarks arrays.
        self._buffer: deque[np.ndarray | None] = deque(maxlen=window_size)
        self._frame_count = 0
        self._last_results: dict[str, GestureResult] | None = None

        # Determine device.
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        logger.info("Gesture classifier using device: %s", self._device)

        # Load trained models.
        self._models: dict[str, torch.nn.Module] = {}
        self._configs: dict[str, dict] = {}
        self._graph = MediaPipeGraph()

        self._load_models(Path(model_dir))

    def _load_models(self, model_dir: Path) -> None:
        """Load all trained ST-GCN models from the model directory.

        Looks for files matching the pattern <gesture_type>.pt with
        a companion <gesture_type>_config.json.

        Args:
            model_dir: Path to the model directory.
        """
        if not model_dir.exists():
            logger.warning(
                "Model directory '%s' does not exist. "
                "Classification will be disabled.",
                model_dir,
            )
            return

        for gesture_type in GESTURE_TYPES:
            model_path = model_dir / f"{gesture_type}.pt"
            config_path = model_dir / f"{gesture_type}_config.json"

            if not model_path.exists():
                logger.debug(
                    "No model found for gesture '%s' at '%s'.",
                    gesture_type,
                    model_path,
                )
                continue

            # Load config if available.
            config = {}
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)

            # Build and load the model.
            num_classes = GESTURE_TYPES[gesture_type]["num_classes"]
            in_channels = config.get("in_channels", 3)

            model = STGCN(
                num_classes=num_classes,
                graph=self._graph,
                in_channels=in_channels,
            )

            state_dict = torch.load(
                model_path,
                map_location=self._device,
                weights_only=True,
            )
            model.load_state_dict(state_dict)
            model.to(self._device)
            model.eval()

            self._models[gesture_type] = model
            self._configs[gesture_type] = config

            logger.info(
                "Loaded gesture model: %s (%s)",
                gesture_type,
                model_path,
            )

        if not self._models:
            logger.warning(
                "No gesture models loaded from '%s'. "
                "Run training first: python -m training.train",
                model_dir,
            )

    @property
    def is_ready(self) -> bool:
        """True if at least one model is loaded."""
        return len(self._models) > 0

    @property
    def has_full_buffer(self) -> bool:
        """True if the buffer has enough frames for classification."""
        return len(self._buffer) >= self.window_size

    def update(
        self, frame_data: FrameLandmarks,
    ) -> dict[str, GestureResult] | None:
        """Add a frame and optionally run classification.

        Adds the frame's world landmarks to the rolling buffer. Once
        the buffer is full, runs inference every `stride` frames.

        Args:
            frame_data: Pose landmarks for the current frame.

        Returns:
            Dict of gesture results if inference was run, or the most
            recent results if available, or None if the buffer isn't
            full yet or no models are loaded.
        """
        if not self.is_ready:
            return None

        # Store world_landmarks (or None for missing detections).
        if frame_data.is_detected:
            self._buffer.append(frame_data.world_landmarks.copy())
        else:
            self._buffer.append(None)

        self._frame_count += 1

        # Run inference when buffer is full and at stride intervals.
        if self.has_full_buffer and self._frame_count % self.stride == 0:
            self._last_results = self._classify()

        return self._last_results

    def _classify(self) -> dict[str, GestureResult]:
        """Run ST-GCN inference on the current buffer contents.

        Returns:
            Dict mapping gesture type names to GestureResult objects.
        """
        input_tensor = self._prepare_input()
        results: dict[str, GestureResult] = {}

        for gesture_type, model in self._models.items():
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()

            gesture_info = GESTURE_TYPES[gesture_type]
            is_positive = pred_class == 1

            if is_positive:
                label = gesture_info["positive_label"].value
            else:
                label = gesture_info["negative_label"].value

            results[gesture_type] = GestureResult(
                gesture_type=gesture_type,
                label=label,
                confidence=confidence,
                is_positive=is_positive,
            )

        return results

    def _prepare_input(self) -> torch.Tensor:
        """Convert the rolling buffer to an ST-GCN input tensor.

        Handles missing detections via linear interpolation or
        zero-fill. Normalizes coordinates relative to the hip
        midpoint for position invariance.

        Returns:
            Tensor of shape (1, 3, T, 33, 1) on the target device.
        """
        buffer_list = list(self._buffer)
        T = len(buffer_list)
        V = 33
        C = 3

        # Build the raw landmark array (T, V, C).
        raw = np.zeros((T, V, C), dtype=np.float32)
        valid_mask = np.zeros(T, dtype=bool)

        for t, wl in enumerate(buffer_list):
            if wl is not None:
                raw[t] = wl[:, :3]  # (33, 3)
                valid_mask[t] = True

        # Interpolate missing frames.
        raw = self._interpolate_missing(raw, valid_mask)

        # Normalize: subtract hip midpoint per frame.
        # Hip midpoint = average of left_hip (23) and right_hip (24).
        hip_midpoint = (raw[:, 23, :] + raw[:, 24, :]) / 2.0  # (T, 3)
        raw = raw - hip_midpoint[:, np.newaxis, :]

        # Reshape to ST-GCN format: (1, C, T, V, M).
        # raw is (T, V, C) → transpose to (C, T, V) → add batch and person dims.
        x = raw.transpose(2, 0, 1)  # (C, T, V)
        x = x[np.newaxis, :, :, :, np.newaxis]  # (1, C, T, V, 1)

        return torch.tensor(x, dtype=torch.float32, device=self._device)

    @staticmethod
    def _interpolate_missing(
        data: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """Linearly interpolate missing frames in the sequence.

        If a frame has no detection (valid_mask=False), interpolate
        from the nearest valid frames on either side. Leading/trailing
        gaps are filled with the nearest valid frame.

        Args:
            data:       Array of shape (T, V, C).
            valid_mask: Boolean array of shape (T,).

        Returns:
            Interpolated data array of shape (T, V, C).
        """
        if valid_mask.all():
            return data

        if not valid_mask.any():
            # No valid frames at all — return zeros.
            return data

        T = data.shape[0]
        valid_indices = np.where(valid_mask)[0]

        for t in range(T):
            if valid_mask[t]:
                continue

            # Find nearest valid frame before and after.
            before = valid_indices[valid_indices < t]
            after = valid_indices[valid_indices > t]

            if len(before) > 0 and len(after) > 0:
                # Interpolate between nearest valid frames.
                t_before = before[-1]
                t_after = after[0]
                alpha = (t - t_before) / (t_after - t_before)
                data[t] = (1 - alpha) * data[t_before] + alpha * data[t_after]
            elif len(before) > 0:
                # No valid frame after — copy last valid.
                data[t] = data[before[-1]]
            else:
                # No valid frame before — copy first valid.
                data[t] = data[after[0]]

        return data
