"""Tests for feedback.gestures — Pepper pose presets."""

from __future__ import annotations

import pytest

from config import PEPPER_JOINTS
from feedback.gestures import (
    GESTURE_POSES,
    POSE_ARMS_CROSSED,
    POSE_ARMS_HIDDEN,
    POSE_NEUTRAL,
    POSE_NOT_FACING,
)


# All poses to test.
_ALL_POSES = {
    "neutral": POSE_NEUTRAL,
    "not_facing": POSE_NOT_FACING,
    "arms_crossed": POSE_ARMS_CROSSED,
    "arms_hidden": POSE_ARMS_HIDDEN,
}


class TestGesturePoses:
    """Tests for the predefined Pepper pose presets."""

    @pytest.mark.parametrize("pose_name,pose", list(_ALL_POSES.items()))
    def test_all_joints_within_limits(
        self,
        pose_name: str,
        pose: dict[str, float],
    ) -> None:
        """Every joint angle in every pose must be within Pepper's limits."""
        for joint_name, angle in pose.items():
            assert joint_name in PEPPER_JOINTS, (
                f"Pose '{pose_name}' references unknown joint '{joint_name}'"
            )
            min_limit, max_limit = PEPPER_JOINTS[joint_name]
            assert min_limit <= angle <= max_limit, (
                f"Pose '{pose_name}', joint '{joint_name}': "
                f"angle {angle:.3f} outside [{min_limit:.3f}, {max_limit:.3f}]"
            )

    @pytest.mark.parametrize("pose_name,pose", list(_ALL_POSES.items()))
    def test_pose_covers_all_joints(
        self,
        pose_name: str,
        pose: dict[str, float],
    ) -> None:
        """Each pose should define all controllable Pepper joints."""
        for joint_name in PEPPER_JOINTS:
            assert joint_name in pose, (
                f"Pose '{pose_name}' is missing joint '{joint_name}'"
            )

    def test_gesture_poses_lookup_has_all_gestures(self) -> None:
        """The GESTURE_POSES lookup table should cover all demo gestures."""
        expected = {"not_facing", "arms_crossed", "arms_hidden", "neutral"}
        assert set(GESTURE_POSES.keys()) == expected

    def test_not_facing_has_significant_head_yaw(self) -> None:
        """The not-facing pose should have noticeable head rotation."""
        assert abs(POSE_NOT_FACING["HeadYaw"]) >= 0.5

    def test_arms_crossed_has_bent_elbows(self) -> None:
        """The arms-crossed pose should have tightly bent elbows."""
        assert abs(POSE_ARMS_CROSSED["LElbowRoll"]) >= 1.0
        assert abs(POSE_ARMS_CROSSED["RElbowRoll"]) >= 1.0

    def test_arms_hidden_has_backward_pitch(self) -> None:
        """The arms-hidden pose should have shoulders pitched backward."""
        assert POSE_ARMS_HIDDEN["LShoulderPitch"] >= 1.5
        assert POSE_ARMS_HIDDEN["RShoulderPitch"] >= 1.5
