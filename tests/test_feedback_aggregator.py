"""Tests for feedback.aggregator — FeedbackAggregator and FeedbackReport."""

from __future__ import annotations

import pytest

from classification.gesture_classifier import GestureResult
from feedback.aggregator import FeedbackAggregator, FeedbackReport


# ---------------------------------------------------------------------------
# Helpers to build mock GestureResult dicts.
# ---------------------------------------------------------------------------

def _make_results(
    facing_positive: bool = True,
    arms_crossed_positive: bool = False,
    arms_hidden_positive: bool = False,
) -> dict[str, GestureResult]:
    """Build a gesture_results dict with the specified states.

    Args:
        facing_positive:      If True, presenter IS facing audience.
        arms_crossed_positive: If True, arms ARE crossed (bad).
        arms_hidden_positive:  If True, arms ARE hidden (bad).

    Returns:
        Dict matching GestureClassifier.update() output format.
    """
    return {
        "facing": GestureResult(
            gesture_type="facing",
            label="facing_audience" if facing_positive else "not_facing",
            confidence=0.9,
            is_positive=facing_positive,
        ),
        "arms_crossed": GestureResult(
            gesture_type="arms_crossed",
            label="arms_crossed" if arms_crossed_positive else "neutral",
            confidence=0.85,
            is_positive=arms_crossed_positive,
        ),
        "arms_hidden": GestureResult(
            gesture_type="arms_hidden",
            label="arms_hidden" if arms_hidden_positive else "neutral",
            confidence=0.88,
            is_positive=arms_hidden_positive,
        ),
    }


# ---------------------------------------------------------------------------
# FeedbackReport tests.
# ---------------------------------------------------------------------------

class TestFeedbackReport:
    """Tests for the FeedbackReport dataclass."""

    def test_has_issues_when_all_zero(self) -> None:
        report = FeedbackReport()
        assert not report.has_issues

    def test_has_issues_when_not_facing(self) -> None:
        report = FeedbackReport(not_facing_count=3)
        assert report.has_issues

    def test_has_issues_when_arms_crossed(self) -> None:
        report = FeedbackReport(arms_crossed_count=1)
        assert report.has_issues

    def test_has_issues_when_arms_hidden(self) -> None:
        report = FeedbackReport(arms_hidden_count=2)
        assert report.has_issues

    def test_issues_sorted_by_count(self) -> None:
        report = FeedbackReport(
            not_facing_count=2,
            arms_crossed_count=5,
            arms_hidden_count=1,
        )
        issues = report.issues
        assert len(issues) == 3
        # Most frequent first.
        assert issues[0] == ("arms_crossed", 5)
        assert issues[1] == ("not_facing", 2)
        assert issues[2] == ("arms_hidden", 1)

    def test_issues_only_nonzero(self) -> None:
        report = FeedbackReport(not_facing_count=3, arms_hidden_count=0)
        issues = report.issues
        assert len(issues) == 1
        assert issues[0][0] == "not_facing"


# ---------------------------------------------------------------------------
# FeedbackAggregator tests.
# ---------------------------------------------------------------------------

class TestFeedbackAggregator:
    """Tests for the FeedbackAggregator."""

    def test_empty_report(self) -> None:
        """No updates → all zeros."""
        agg = FeedbackAggregator()
        report = agg.build_report()
        assert report.not_facing_count == 0
        assert report.arms_crossed_count == 0
        assert report.arms_hidden_count == 0
        assert report.total_windows == 0

    def test_none_results_ignored(self) -> None:
        """Passing None should be a no-op."""
        agg = FeedbackAggregator()
        agg.update(None)
        agg.update(None)
        report = agg.build_report()
        assert report.total_windows == 0

    def test_all_positive_no_issues(self) -> None:
        """All-positive results (good posture) → zero counts."""
        agg = FeedbackAggregator()
        good = _make_results(facing_positive=True)
        for _ in range(10):
            agg.update(good)
        report = agg.build_report()
        assert not report.has_issues
        assert report.total_windows == 10

    def test_single_not_facing_transition(self) -> None:
        """One transition to not-facing counts as 1."""
        agg = FeedbackAggregator()
        # 3 frames of good posture.
        for _ in range(3):
            agg.update(_make_results(facing_positive=True))
        # 5 frames of not facing.
        for _ in range(5):
            agg.update(_make_results(facing_positive=False))

        report = agg.build_report()
        assert report.not_facing_count == 1

    def test_multiple_transitions(self) -> None:
        """Multiple on-off transitions count correctly."""
        agg = FeedbackAggregator()
        # Transition 1: good → not facing.
        agg.update(_make_results(facing_positive=True))
        agg.update(_make_results(facing_positive=False))
        # Transition back: not facing → good.
        agg.update(_make_results(facing_positive=True))
        # Transition 2: good → not facing again.
        agg.update(_make_results(facing_positive=False))
        agg.update(_make_results(facing_positive=False))

        report = agg.build_report()
        assert report.not_facing_count == 2

    def test_arms_crossed_counted(self) -> None:
        """Arms crossed (is_positive=True) should count as negative."""
        agg = FeedbackAggregator()
        agg.update(_make_results(arms_crossed_positive=False))
        agg.update(_make_results(arms_crossed_positive=True))  # Transition!
        agg.update(_make_results(arms_crossed_positive=True))  # Stay (no count).

        report = agg.build_report()
        assert report.arms_crossed_count == 1

    def test_arms_hidden_counted(self) -> None:
        """Arms hidden transition is counted."""
        agg = FeedbackAggregator()
        agg.update(_make_results(arms_hidden_positive=True))   # Transition!
        agg.update(_make_results(arms_hidden_positive=False))   # Back.
        agg.update(_make_results(arms_hidden_positive=True))   # Transition 2!

        report = agg.build_report()
        assert report.arms_hidden_count == 2

    def test_multiple_simultaneous_gestures(self) -> None:
        """Multiple gestures active at the same time."""
        agg = FeedbackAggregator()
        # Not facing AND arms crossed at the same time.
        agg.update(_make_results(
            facing_positive=False,
            arms_crossed_positive=True,
        ))
        report = agg.build_report()
        assert report.not_facing_count == 1
        assert report.arms_crossed_count == 1

    def test_reset(self) -> None:
        """Reset clears all state."""
        agg = FeedbackAggregator()
        agg.update(_make_results(facing_positive=False))
        agg.reset()
        report = agg.build_report()
        assert report.total_windows == 0
        assert not report.has_issues

    def test_total_windows_counts_all_updates(self) -> None:
        """total_windows increments for every non-None update."""
        agg = FeedbackAggregator()
        agg.update(_make_results())
        agg.update(None)  # Skipped.
        agg.update(_make_results())
        report = agg.build_report()
        assert report.total_windows == 2
