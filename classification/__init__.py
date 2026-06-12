"""
Gesture classification using ST-GCN (Spatial-Temporal Graph Convolutional Network).

Provides real-time gesture classification from MediaPipe pose landmark
sequences. The ST-GCN model learns spatial (joint relationships) and
temporal (movement over time) patterns directly from the skeleton graph.
"""

from classification.gesture_labels import GestureLabel, GESTURE_TYPES
from classification.gesture_classifier import GestureClassifier, GestureResult

__all__ = [
    "GestureLabel",
    "GestureClassifier",
    "GestureResult",
    "GESTURE_TYPES",
]
