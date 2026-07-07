"""
Predefined Pepper pose presets for gesture demonstration.

Each preset is a dictionary of Pepper joint angles (in radians) that
visually replicates a specific body language gesture. All angles are
within Pepper's documented physical joint limits defined in config.py.

These poses are used by the PepperFeedbackController to demonstrate
what negative body language looks like before explaining it verbally.
"""

from __future__ import annotations

from config import PEPPER_JOINTS
from utils.angle_utils import clamp_angle


def _clamp_pose(pose: dict[str, float]) -> dict[str, float]:
    """Clamp all angles in a pose dict to Pepper's physical limits.

    Args:
        pose: Dictionary of joint_name → angle (radians).

    Returns:
        New dictionary with all angles clamped.
    """
    clamped = {}
    for joint, angle in pose.items():
        if joint in PEPPER_JOINTS:
            min_limit, max_limit = PEPPER_JOINTS[joint]
            clamped[joint] = clamp_angle(angle, min_limit, max_limit)
        else:
            clamped[joint] = angle
    return clamped


# ---------------------------------------------------------------------------
# Neutral / rest pose — all joints at 0 (Pepper's default standing pose).
# ---------------------------------------------------------------------------

POSE_NEUTRAL: dict[str, float] = _clamp_pose({
    "HeadYaw":        0.0,
    "HeadPitch":      0.0,
    "LShoulderPitch": 1.4,    # Arms relaxed at sides.
    "LShoulderRoll":  0.15,
    "LElbowYaw":      -1.2,
    "LElbowRoll":     -0.5,
    "LWristYaw":      0.0,
    "RShoulderPitch": 1.4,
    "RShoulderRoll":  -0.15,
    "RElbowYaw":      1.2,
    "RElbowRoll":     0.5,
    "RWristYaw":      0.0,
})


# ---------------------------------------------------------------------------
# Not facing the audience — Pepper turns head and torso to one side.
# ---------------------------------------------------------------------------

POSE_NOT_FACING: dict[str, float] = _clamp_pose({
    "HeadYaw":        1.0,    # Head turned ~60° to the right.
    "HeadPitch":      0.1,    # Slightly looking down.
    "LShoulderPitch": 1.4,
    "LShoulderRoll":  0.15,
    "LElbowYaw":      -1.2,
    "LElbowRoll":     -0.5,
    "LWristYaw":      0.0,
    "RShoulderPitch": 1.4,
    "RShoulderRoll":  -0.15,
    "RElbowYaw":      1.2,
    "RElbowRoll":     0.5,
    "RWristYaw":      0.0,
})


# ---------------------------------------------------------------------------
# Arms crossed — both arms folded across the chest.
# ---------------------------------------------------------------------------

POSE_ARMS_CROSSED: dict[str, float] = _clamp_pose({
    "HeadYaw":        0.0,
    "HeadPitch":      0.1,
    # Left arm: shoulder forward, elbow bent sharply inward.
    "LShoulderPitch": 0.4,    # Arm pitched forward.
    "LShoulderRoll":  0.05,   # Arm close to body.
    "LElbowYaw":      -0.3,   # Forearm angled across chest.
    "LElbowRoll":     -1.4,   # Elbow bent tightly.
    "LWristYaw":      0.5,
    # Right arm: mirrors left, crossing over.
    "RShoulderPitch": 0.4,
    "RShoulderRoll":  -0.05,
    "RElbowYaw":      0.3,
    "RElbowRoll":     1.4,
    "RWristYaw":      -0.5,
})


# ---------------------------------------------------------------------------
# Arms hidden — arms tucked behind the back.
# ---------------------------------------------------------------------------

POSE_ARMS_HIDDEN: dict[str, float] = _clamp_pose({
    "HeadYaw":        0.0,
    "HeadPitch":      0.0,
    # Left arm: pitched backward, elbow bent behind torso.
    "LShoulderPitch": 1.8,    # Arm pitched backward / behind.
    "LShoulderRoll":  0.10,   # Arm close to body.
    "LElbowYaw":      -1.8,   # Forearm twisted behind.
    "LElbowRoll":     -0.5,   # Slight bend.
    "LWristYaw":      0.0,
    # Right arm: mirrors left.
    "RShoulderPitch": 1.8,
    "RShoulderRoll":  -0.10,
    "RElbowYaw":      1.8,
    "RElbowRoll":     0.5,
    "RWristYaw":      0.0,
})


# ---------------------------------------------------------------------------
# Lookup table for the controller to access poses by gesture key.
# ---------------------------------------------------------------------------

GESTURE_POSES: dict[str, dict[str, float]] = {
    "not_facing":   POSE_NOT_FACING,
    "arms_crossed": POSE_ARMS_CROSSED,
    "arms_hidden":  POSE_ARMS_HIDDEN,
    "neutral":      POSE_NEUTRAL,
}
