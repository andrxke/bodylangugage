"""
Gesture taxonomy and multi-label support.

Defines the gesture types that the pipeline can classify. Each gesture
type is treated as an independent binary classification task, so
multiple gestures can be active simultaneously (e.g., facing the audience
AND arms crossed).
"""

from __future__ import annotations

from enum import Enum


class GestureLabel(str, Enum):
    """All possible gesture labels across all gesture types."""

    FACING_AUDIENCE = "facing_audience"
    NOT_FACING = "not_facing"
    ARMS_CROSSED = "arms_crossed"
    NEUTRAL = "neutral"
    ARMS_HIDDEN = "arms_hidden"


# Each gesture type is a binary classification task.
# The CSV manifest uses the 'csv_column' name as the column header.
# Positive class = the gesture is present, negative = it is not.
GESTURE_TYPES: dict[str, dict] = {
    "facing": {
        "csv_column": "facing",
        "positive_label": GestureLabel.FACING_AUDIENCE,
        "negative_label": GestureLabel.NOT_FACING,
        "description": "Is the presenter facing the audience?",
        "num_classes": 2,
    },
    "arms_crossed": {
        "csv_column": "arms_crossed",
        "positive_label": GestureLabel.ARMS_CROSSED,
        "negative_label": GestureLabel.NEUTRAL,
        "description": "Are the presenter's arms crossed?",
        "num_classes": 2,
    },
    "arms_hidden": {
        "csv_column": "arms_hidden",
        "positive_label": GestureLabel.ARMS_HIDDEN,
        "negative_label": GestureLabel.NEUTRAL,
        "description": "Are the presenter's arms hidden?",
        "num_classes": 2,
    },
}
