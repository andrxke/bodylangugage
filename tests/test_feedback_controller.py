"""Tests for feedback.controller — PepperFeedbackController."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from feedback.aggregator import FeedbackReport
from feedback.controller import (
    PepperFeedbackController,
    _CONGRATULATIONS_MESSAGE,
    _FEEDBACK_MESSAGES,
    _INTRO_MESSAGE,
)
from feedback.gestures import GESTURE_POSES, POSE_NEUTRAL
from feedback.motion import MotionDriver
from feedback.speech import SpeechDriver


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------

class MockMotionDriver(MotionDriver):
    """Mock motion driver that records calls."""

    def __init__(self) -> None:
        self.poses: list[tuple[dict[str, float], float]] = []
        self.neutrals: list[float] = []
        self.closed = False

    def set_pose(self, angles: dict[str, float], speed: float = 0.2) -> None:
        self.poses.append((angles, speed))

    def go_neutral(self, speed: float = 0.2) -> None:
        self.neutrals.append(speed)

    def close(self) -> None:
        self.closed = True


class MockSpeechDriver(SpeechDriver):
    """Mock speech driver that records calls."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.closed = False

    def say(self, text: str) -> None:
        self.spoken.append(text)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def mock_motion() -> MockMotionDriver:
    return MockMotionDriver()


@pytest.fixture
def mock_speech() -> MockSpeechDriver:
    return MockSpeechDriver()


@pytest.fixture
def controller(
    mock_motion: MockMotionDriver,
    mock_speech: MockSpeechDriver,
) -> PepperFeedbackController:
    return PepperFeedbackController(
        motion_driver=mock_motion,
        speech_driver=mock_speech,
        pose_settle_time=0.0,  # No sleep in tests.
        speech_pause=0.0,
        motion_speed=0.2,
    )


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------

class TestPepperFeedbackController:
    """Tests for the feedback controller orchestration."""

    @patch("feedback.controller.time.sleep")
    def test_no_issues_congratulates(
        self,
        mock_sleep: MagicMock,
        controller: PepperFeedbackController,
        mock_speech: MockSpeechDriver,
        mock_motion: MockMotionDriver,
    ) -> None:
        """No issues → congratulatory message, no poses."""
        report = FeedbackReport(total_windows=10)
        controller.deliver_feedback(report)

        assert len(mock_speech.spoken) == 1
        assert mock_speech.spoken[0] == _CONGRATULATIONS_MESSAGE
        assert len(mock_motion.poses) == 0

    @patch("feedback.controller.time.sleep")
    def test_single_issue_demonstrated(
        self,
        mock_sleep: MagicMock,
        controller: PepperFeedbackController,
        mock_speech: MockSpeechDriver,
        mock_motion: MockMotionDriver,
    ) -> None:
        """One issue → intro + demo pose + speech + neutral."""
        report = FeedbackReport(not_facing_count=4, total_windows=20)
        controller.deliver_feedback(report)

        # Speech: intro + feedback message.
        assert len(mock_speech.spoken) == 2
        assert mock_speech.spoken[0] == _INTRO_MESSAGE
        assert "4" in mock_speech.spoken[1]
        assert "face the audience" in mock_speech.spoken[1]

        # Motion: demo pose set once.
        assert len(mock_motion.poses) == 1
        assert mock_motion.poses[0][0] == GESTURE_POSES["not_facing"]

        # Neutral called: once after demo + once at the end.
        assert len(mock_motion.neutrals) == 2

    @patch("feedback.controller.time.sleep")
    def test_multiple_issues_all_demonstrated(
        self,
        mock_sleep: MagicMock,
        controller: PepperFeedbackController,
        mock_speech: MockSpeechDriver,
        mock_motion: MockMotionDriver,
    ) -> None:
        """Multiple issues → each gets a demo, ordered by count."""
        report = FeedbackReport(
            not_facing_count=2,
            arms_crossed_count=5,
            arms_hidden_count=1,
            total_windows=30,
        )
        controller.deliver_feedback(report)

        # Speech: intro + 3 feedback messages.
        assert len(mock_speech.spoken) == 4

        # First demo should be the most frequent (arms_crossed=5).
        assert "5" in mock_speech.spoken[1]
        assert "crossed" in mock_speech.spoken[1]

        # Second: not_facing=2.
        assert "2" in mock_speech.spoken[2]
        assert "face" in mock_speech.spoken[2]

        # Third: arms_hidden=1.
        assert "1" in mock_speech.spoken[3]
        assert "hid" in mock_speech.spoken[3]

        # 3 demo poses were set.
        assert len(mock_motion.poses) == 3

    @patch("feedback.controller.time.sleep")
    def test_singular_time_in_message(
        self,
        mock_sleep: MagicMock,
        controller: PepperFeedbackController,
        mock_speech: MockSpeechDriver,
    ) -> None:
        """Count of 1 should use 'time' not 'times'."""
        report = FeedbackReport(arms_crossed_count=1, total_windows=5)
        controller.deliver_feedback(report)

        # The feedback message for count=1 should say "1 time".
        assert "1 time " in mock_speech.spoken[1]
        assert "1 times" not in mock_speech.spoken[1]

    @patch("feedback.controller.time.sleep")
    def test_plural_times_in_message(
        self,
        mock_sleep: MagicMock,
        controller: PepperFeedbackController,
        mock_speech: MockSpeechDriver,
    ) -> None:
        """Count > 1 should use 'times'."""
        report = FeedbackReport(arms_crossed_count=3, total_windows=5)
        controller.deliver_feedback(report)

        assert "3 times" in mock_speech.spoken[1]

    @patch("feedback.controller.time.sleep")
    def test_close_cleans_up_drivers(
        self,
        mock_sleep: MagicMock,
        controller: PepperFeedbackController,
        mock_motion: MockMotionDriver,
        mock_speech: MockSpeechDriver,
    ) -> None:
        """close() should call close on both drivers."""
        controller.close()
        assert mock_motion.closed
        assert mock_speech.closed

    @patch("feedback.controller.time.sleep")
    def test_unknown_gesture_skipped(
        self,
        mock_sleep: MagicMock,
        controller: PepperFeedbackController,
        mock_motion: MockMotionDriver,
        mock_speech: MockSpeechDriver,
    ) -> None:
        """A gesture key with no pose should be skipped gracefully."""
        # Call the private method directly with an unknown key.
        controller._demonstrate_gesture("unknown_gesture", 3)
        assert len(mock_motion.poses) == 0
        assert len(mock_speech.spoken) == 0
