"""
Pepper feedback controller — orchestrates gesture demonstration.

Ties together the motion driver, speech driver, and gesture poses to
deliver a complete post-session feedback sequence where Pepper:

  1. Demonstrates each detected negative body language gesture
  2. Verbally explains the observation with specific counts
  3. Returns to a neutral pose between demonstrations

This module is deliberately decoupled from the capture/classification
pipeline so it can be swapped, extended, or replaced independently.

Usage:
    controller = PepperFeedbackController(
        motion_driver=motion,
        speech_driver=speech,
        pose_settle_time=2.0,
        speech_pause=1.5,
    )
    controller.deliver_feedback(report)
    controller.close()
"""

from __future__ import annotations

import logging
import time

from feedback.aggregator import FeedbackReport
from feedback.gestures import GESTURE_POSES, POSE_NEUTRAL
from feedback.motion import MotionDriver
from feedback.speech import SpeechDriver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feedback message templates.
# {count} is replaced with the number of occurrences.
# ---------------------------------------------------------------------------

_FEEDBACK_MESSAGES: dict[str, str] = {
    "not_facing": (
        "I noticed you didn't face the audience {count} "
        "{times} during the presentation."
    ),
    "arms_crossed": (
        "I noticed you crossed your arms {count} "
        "{times} during the presentation."
    ),
    "arms_hidden": (
        "I noticed you hid your arms behind your back {count} "
        "{times} during the presentation."
    ),
}

_CONGRATULATIONS_MESSAGE = (
    "Great job! I didn't detect any negative body language "
    "during your presentation. Keep up the good work!"
)

_INTRO_MESSAGE = (
    "Let me show you some body language I noticed during your presentation."
)


def _pluralize_times(count: int) -> str:
    """Return 'time' or 'times' based on count.

    Args:
        count: The number of occurrences.

    Returns:
        'time' if count == 1, 'times' otherwise.
    """
    return "time" if count == 1 else "times"


class PepperFeedbackController:
    """Orchestrates post-session feedback delivery on Pepper.

    For each detected negative gesture, Pepper:
      1. Moves into the demonstration pose.
      2. Waits for the pose to settle.
      3. Speaks the feedback message with the occurrence count.
      4. Returns to the neutral pose.
      5. Pauses briefly before the next demonstration.

    If no issues were detected, delivers a congratulatory message.

    Attributes:
        pose_settle_time: Seconds to hold each demo pose before speaking.
        speech_pause:     Seconds to pause after speech before next gesture.
        motion_speed:     Joint movement speed [0.0, 1.0].
    """

    def __init__(
        self,
        motion_driver: MotionDriver,
        speech_driver: SpeechDriver,
        pose_settle_time: float = 2.0,
        speech_pause: float = 1.5,
        motion_speed: float = 0.2,
    ) -> None:
        """Initialize the feedback controller.

        Args:
            motion_driver:    Driver for controlling Pepper's joints.
            speech_driver:    Driver for Pepper's text-to-speech.
            pose_settle_time: Seconds to hold each demo pose.
            speech_pause:     Seconds to pause after speech.
            motion_speed:     Joint movement speed [0.0, 1.0].
        """
        self._motion = motion_driver
        self._speech = speech_driver
        self.pose_settle_time = pose_settle_time
        self.speech_pause = speech_pause
        self.motion_speed = motion_speed

    def deliver_feedback(self, report: FeedbackReport) -> None:
        """Deliver the full feedback sequence based on the report.

        Args:
            report: FeedbackReport with gesture occurrence counts.
        """
        logger.info("Starting Pepper feedback delivery...")

        if not report.has_issues:
            logger.info("No negative gestures detected — congratulating.")
            self._speech.say(_CONGRATULATIONS_MESSAGE)
            return

        # Introduction.
        self._speech.say(_INTRO_MESSAGE)
        time.sleep(self.speech_pause)

        # Demonstrate each issue (sorted by count, most frequent first).
        for gesture_key, count in report.issues:
            self._demonstrate_gesture(gesture_key, count)

        # Return to neutral at the end.
        self._motion.go_neutral(self.motion_speed)

        logger.info("Pepper feedback delivery complete.")

    def _demonstrate_gesture(self, gesture_key: str, count: int) -> None:
        """Demonstrate a single gesture with pose and speech.

        Args:
            gesture_key: Key into GESTURE_POSES (e.g. "not_facing").
            count:       Number of times the gesture was detected.
        """
        logger.info(
            "Demonstrating gesture '%s' (count=%d)", gesture_key, count,
        )

        # 1. Move into the demonstration pose.
        pose = GESTURE_POSES.get(gesture_key)
        if pose is None:
            logger.warning(
                "No pose defined for gesture '%s', skipping.", gesture_key,
            )
            return

        self._motion.set_pose(pose, self.motion_speed)
        time.sleep(self.pose_settle_time)

        # 2. Speak the feedback message.
        message_template = _FEEDBACK_MESSAGES.get(gesture_key)
        if message_template is not None:
            message = message_template.format(
                count=count,
                times=_pluralize_times(count),
            )
            self._speech.say(message)
        else:
            logger.warning(
                "No feedback message template for '%s'.", gesture_key,
            )

        time.sleep(self.speech_pause)

        # 3. Return to neutral before the next demonstration.
        self._motion.go_neutral(self.motion_speed)
        time.sleep(1.0)  # Brief pause between demonstrations.

    def close(self) -> None:
        """Clean up drivers."""
        self._motion.close()
        self._speech.close()
        logger.info("Feedback controller closed.")
