"""
Feedback aggregator — accumulates gesture classification results.

Tracks how many times each negative body language gesture was detected
across a session and produces a summary report. Uses state-transition
counting so that consecutive frames of the same gesture only count as
a single occurrence.

Usage:
    aggregator = FeedbackAggregator()

    # In the frame loop:
    aggregator.update(gesture_results)

    # After the session:
    report = aggregator.build_report()
    print(report.not_facing_count)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from classification.gesture_classifier import GestureResult

logger = logging.getLogger(__name__)


@dataclass
class FeedbackReport:
    """Summary of detected body language issues over a session.

    Counts represent distinct occurrences (state transitions to a
    negative gesture), not raw frame counts.

    Attributes:
        not_facing_count:   Times the presenter turned away from the audience.
        arms_crossed_count: Times the presenter crossed their arms.
        arms_hidden_count:  Times the presenter hid their arms.
        total_windows:      Total number of classification windows processed.
    """

    not_facing_count: int = 0
    arms_crossed_count: int = 0
    arms_hidden_count: int = 0
    total_windows: int = 0

    @property
    def has_issues(self) -> bool:
        """True if any negative gestures were detected."""
        return (
            self.not_facing_count > 0
            or self.arms_crossed_count > 0
            or self.arms_hidden_count > 0
        )

    @property
    def issues(self) -> list[tuple[str, int]]:
        """Return a list of (gesture_key, count) for non-zero issues.

        Sorted by count descending so the most frequent issue comes first.
        """
        items = []
        if self.not_facing_count > 0:
            items.append(("not_facing", self.not_facing_count))
        if self.arms_crossed_count > 0:
            items.append(("arms_crossed", self.arms_crossed_count))
        if self.arms_hidden_count > 0:
            items.append(("arms_hidden", self.arms_hidden_count))

        items.sort(key=lambda x: x[1], reverse=True)
        return items


# Mapping from gesture type key to the attribute name on FeedbackReport
# and the "negative" predicate (is_positive=False means the bad gesture
# is happening).
_GESTURE_TRACKING = {
    "facing": {
        "report_attr": "not_facing_count",
        "is_negative": lambda r: not r.is_positive,  # not facing = negative
    },
    "arms_crossed": {
        "report_attr": "arms_crossed_count",
        "is_negative": lambda r: r.is_positive,  # arms_crossed = positive class IS negative behavior
    },
    "arms_hidden": {
        "report_attr": "arms_hidden_count",
        "is_negative": lambda r: r.is_positive,  # arms_hidden = positive class IS negative behavior
    },
}


class FeedbackAggregator:
    """Accumulates per-frame gesture results into a session summary.

    Uses state-transition counting: a gesture occurrence is counted
    only when the state *transitions* from non-negative to negative.
    This ensures that a presenter who turns away for 30 consecutive
    frames counts as 1 occurrence, not 30.

    Attributes:
        total_windows: Number of classification windows processed.
    """

    def __init__(self) -> None:
        """Initialize the aggregator with zero counts."""
        self._counts: dict[str, int] = {
            info["report_attr"]: 0 for info in _GESTURE_TRACKING.values()
        }
        self._prev_negative: dict[str, bool] = {
            key: False for key in _GESTURE_TRACKING
        }
        self.total_windows: int = 0

    def update(self, gesture_results: dict[str, GestureResult] | None) -> None:
        """Process one frame's gesture classification results.

        Args:
            gesture_results: Dict of gesture type → GestureResult from
                             GestureClassifier.update(), or None if
                             classification hasn't run yet.
        """
        if gesture_results is None:
            return

        self.total_windows += 1

        for gesture_key, info in _GESTURE_TRACKING.items():
            if gesture_key not in gesture_results:
                continue

            result = gesture_results[gesture_key]
            is_negative_now = info["is_negative"](result)
            was_negative = self._prev_negative[gesture_key]

            # Count only on transition: not-negative → negative.
            if is_negative_now and not was_negative:
                self._counts[info["report_attr"]] += 1

            self._prev_negative[gesture_key] = is_negative_now

    def build_report(self) -> FeedbackReport:
        """Build the final feedback report from accumulated data.

        Returns:
            FeedbackReport with counts and totals.
        """
        report = FeedbackReport(
            not_facing_count=self._counts["not_facing_count"],
            arms_crossed_count=self._counts["arms_crossed_count"],
            arms_hidden_count=self._counts["arms_hidden_count"],
            total_windows=self.total_windows,
        )

        logger.info(
            "Feedback report: not_facing=%d, arms_crossed=%d, "
            "arms_hidden=%d (from %d windows)",
            report.not_facing_count,
            report.arms_crossed_count,
            report.arms_hidden_count,
            report.total_windows,
        )

        return report

    def reset(self) -> None:
        """Reset all counts and state for a new session."""
        for key in self._counts:
            self._counts[key] = 0
        for key in self._prev_negative:
            self._prev_negative[key] = False
        self.total_windows = 0
