"""
Pepper robot joint angle mapping from MediaPipe pose landmarks.

Converts the 33-landmark world-coordinate skeleton from MediaPipe's
PoseLandmarker into the joint angles that Pepper's NAOqi ALMotion API
expects. Each angle is computed via 3D vector geometry and clamped to
Pepper's documented physical joint limits.

Pepper joint conventions:
  - Pitch: rotation around the lateral (Y) axis (nodding, arm up/down)
  - Roll:  rotation around the longitudinal (X) axis (tilting, arm in/out)
  - Yaw:   rotation around the vertical (Z) axis (turning, forearm twist)

Usage (future — when Pepper controller module is implemented):
    angles = compute_pepper_angles(frame.world_landmarks)
    pepper_motion.setAngles(list(angles.keys()), list(angles.values()), 0.2)
"""

from __future__ import annotations

import numpy as np

from config import PEPPER_JOINTS, LANDMARK_INDICES
from utils.angle_utils import (
    calculate_angle,
    clamp_angle,
    rotation_angle_around_axis,
    vector_between,
)


# ---------------------------------------------------------------------------
# Convenience aliases for landmark indices used in angle calculations.
# ---------------------------------------------------------------------------
_NOSE = LANDMARK_INDICES["nose"]
_L_EAR = LANDMARK_INDICES["left_ear"]
_R_EAR = LANDMARK_INDICES["right_ear"]
_L_SHOULDER = LANDMARK_INDICES["left_shoulder"]
_R_SHOULDER = LANDMARK_INDICES["right_shoulder"]
_L_ELBOW = LANDMARK_INDICES["left_elbow"]
_R_ELBOW = LANDMARK_INDICES["right_elbow"]
_L_WRIST = LANDMARK_INDICES["left_wrist"]
_R_WRIST = LANDMARK_INDICES["right_wrist"]
_L_HIP = LANDMARK_INDICES["left_hip"]
_R_HIP = LANDMARK_INDICES["right_hip"]
_L_INDEX = LANDMARK_INDICES["left_index"]
_R_INDEX = LANDMARK_INDICES["right_index"]


def _get_point(world_landmarks: np.ndarray, index: int) -> np.ndarray:
    """Extract a 3D point (x, y, z) from the world landmarks array.

    Args:
        world_landmarks: Shape (33, 3) array of world coordinates.
        index:           Landmark index (0–32).

    Returns:
        1D array of shape (3,).
    """
    return world_landmarks[index]


# ---------------------------------------------------------------------------
# Individual joint angle calculators.
# Each function takes the full (33, 3) world_landmarks array and returns
# a single angle in radians.
# ---------------------------------------------------------------------------

def _head_yaw(wl: np.ndarray) -> float:
    """Compute Pepper HeadYaw from nose and ear positions.

    HeadYaw is the horizontal rotation of the head. Positive = turned left.
    We measure it as the signed angle between the torso's forward direction
    and the nose direction, projected onto the horizontal plane.
    """
    # Torso forward: perpendicular to the shoulder line, pointing forward.
    l_shoulder = _get_point(wl, _L_SHOULDER)
    r_shoulder = _get_point(wl, _R_SHOULDER)
    mid_shoulder = (l_shoulder + r_shoulder) / 2.0

    # Shoulder line vector (left to right).
    shoulder_vec = vector_between(l_shoulder, r_shoulder)

    # Torso forward direction (cross product of shoulder line with up vector).
    up = np.array([0.0, -1.0, 0.0])  # MediaPipe Y points down.
    torso_forward = np.cross(shoulder_vec, up)

    # Nose direction from shoulder midpoint.
    nose = _get_point(wl, _NOSE)
    nose_dir = vector_between(mid_shoulder, nose)

    # Yaw = rotation around the vertical axis.
    return rotation_angle_around_axis(torso_forward, nose_dir, up)


def _head_pitch(wl: np.ndarray) -> float:
    """Compute Pepper HeadPitch from nose and shoulder positions.

    HeadPitch is the vertical tilt of the head. Negative = looking up.
    Measured as the angle between the shoulder-to-nose vector and horizontal.
    """
    l_shoulder = _get_point(wl, _L_SHOULDER)
    r_shoulder = _get_point(wl, _R_SHOULDER)
    mid_shoulder = (l_shoulder + r_shoulder) / 2.0
    nose = _get_point(wl, _NOSE)

    # Vector from shoulder midpoint to nose.
    neck_to_nose = vector_between(mid_shoulder, nose)

    # Horizontal reference (same vector with y=0).
    horizontal = neck_to_nose.copy()
    horizontal[1] = 0.0

    norm_h = np.linalg.norm(horizontal)
    if norm_h < 1e-8:
        return 0.0

    # Signed pitch: positive when looking down (nose below shoulders).
    angle = calculate_angle(
        mid_shoulder + horizontal,
        mid_shoulder,
        nose,
    )

    # Sign convention: positive = looking down in Pepper's frame.
    if neck_to_nose[1] > 0:  # Y is down in MediaPipe.
        return angle
    return -angle


def _shoulder_pitch(wl: np.ndarray, side: str) -> float:
    """Compute shoulder pitch (arm raised / lowered relative to torso).

    Shoulder pitch is the angle of the upper arm relative to the torso
    in the sagittal plane. Arms down = ~π, arms forward/up = decreasing.

    Args:
        wl:   World landmarks array (33, 3).
        side: 'left' or 'right'.
    """
    if side == "left":
        hip = _get_point(wl, _L_HIP)
        shoulder = _get_point(wl, _L_SHOULDER)
        elbow = _get_point(wl, _L_ELBOW)
    else:
        hip = _get_point(wl, _R_HIP)
        shoulder = _get_point(wl, _R_SHOULDER)
        elbow = _get_point(wl, _R_ELBOW)

    # Angle at the shoulder between the torso (hip→shoulder) and
    # the upper arm (shoulder→elbow).
    angle = calculate_angle(hip, shoulder, elbow)

    # Pepper convention: 0 = arm straight forward, positive = arm down.
    # MediaPipe angle of ~π = arm along torso (down).
    # We map: arms down → ~1.5 rad, arms up → ~-2.0 rad.
    return angle - np.pi / 2.0


def _shoulder_roll(wl: np.ndarray, side: str) -> float:
    """Compute shoulder roll (arm abducted outward / adducted inward).

    Shoulder roll is the angle of the upper arm in the coronal plane.

    Args:
        wl:   World landmarks array (33, 3).
        side: 'left' or 'right'.
    """
    if side == "left":
        shoulder = _get_point(wl, _L_SHOULDER)
        elbow = _get_point(wl, _L_ELBOW)
        r_shoulder = _get_point(wl, _R_SHOULDER)
    else:
        shoulder = _get_point(wl, _R_SHOULDER)
        elbow = _get_point(wl, _R_ELBOW)
        r_shoulder = _get_point(wl, _L_SHOULDER)

    # Vector from shoulder to elbow (upper arm direction).
    upper_arm = vector_between(shoulder, elbow)

    # Shoulder line (towards the other shoulder).
    cross_shoulder = vector_between(shoulder, r_shoulder)

    # Torso vertical (approximate — from hip to shoulder).
    if side == "left":
        torso = vector_between(_get_point(wl, _L_HIP), shoulder)
    else:
        torso = vector_between(_get_point(wl, _R_HIP), shoulder)

    # Roll = rotation in the coronal plane (around the forward axis).
    forward = np.cross(cross_shoulder, torso)

    angle = rotation_angle_around_axis(torso, upper_arm, forward)

    # Clamp sign for Pepper's conventions.
    if side == "left":
        return abs(angle)
    return -abs(angle)


def _elbow_roll(wl: np.ndarray, side: str) -> float:
    """Compute elbow flexion angle (elbow roll in Pepper's convention).

    This is the classic elbow bend angle at the elbow joint.

    Args:
        wl:   World landmarks array (33, 3).
        side: 'left' or 'right'.
    """
    if side == "left":
        shoulder = _get_point(wl, _L_SHOULDER)
        elbow = _get_point(wl, _L_ELBOW)
        wrist = _get_point(wl, _L_WRIST)
    else:
        shoulder = _get_point(wl, _R_SHOULDER)
        elbow = _get_point(wl, _R_ELBOW)
        wrist = _get_point(wl, _R_WRIST)

    # Angle at the elbow between upper arm and forearm.
    angle = calculate_angle(shoulder, elbow, wrist)

    # Pepper convention: fully extended = 0, flexed = negative (left)
    # or positive (right).
    flexion = -(np.pi - angle)

    if side == "left":
        return flexion  # Negative range for left elbow.
    return -flexion     # Positive range for right elbow.


def _elbow_yaw(wl: np.ndarray, side: str) -> float:
    """Compute elbow yaw (forearm rotation / pronation-supination).

    This measures the twist of the forearm around the upper-arm axis.

    Args:
        wl:   World landmarks array (33, 3).
        side: 'left' or 'right'.
    """
    if side == "left":
        shoulder = _get_point(wl, _L_SHOULDER)
        elbow = _get_point(wl, _L_ELBOW)
        wrist = _get_point(wl, _L_WRIST)
    else:
        shoulder = _get_point(wl, _R_SHOULDER)
        elbow = _get_point(wl, _R_ELBOW)
        wrist = _get_point(wl, _R_WRIST)

    # Upper arm axis (shoulder → elbow).
    upper_arm = vector_between(shoulder, elbow)

    # Forearm direction (elbow → wrist).
    forearm = vector_between(elbow, wrist)

    # Reference: vertical direction.
    vertical = np.array([0.0, -1.0, 0.0])

    return rotation_angle_around_axis(vertical, forearm, upper_arm)


def _wrist_yaw(wl: np.ndarray, side: str) -> float:
    """Compute wrist yaw (hand rotation relative to forearm).

    Uses the index finger knuckle landmark to estimate hand orientation.

    Args:
        wl:   World landmarks array (33, 3).
        side: 'left' or 'right'.
    """
    if side == "left":
        elbow = _get_point(wl, _L_ELBOW)
        wrist = _get_point(wl, _L_WRIST)
        index = _get_point(wl, _L_INDEX)
    else:
        elbow = _get_point(wl, _R_ELBOW)
        wrist = _get_point(wl, _R_WRIST)
        index = _get_point(wl, _R_INDEX)

    # Forearm axis (elbow → wrist).
    forearm = vector_between(elbow, wrist)

    # Hand direction (wrist → index finger).
    hand = vector_between(wrist, index)

    # Reference: vertical.
    vertical = np.array([0.0, -1.0, 0.0])

    return rotation_angle_around_axis(vertical, hand, forearm)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_pepper_angles(world_landmarks: np.ndarray) -> dict[str, float]:
    """Convert MediaPipe world landmarks to Pepper joint angles.

    Computes all mappable joint angles from the 33-point world-coordinate
    pose skeleton, then clamps each to Pepper's documented physical limits.

    Args:
        world_landmarks: Shape (33, 3) array from PoseLandmarker
                         (real-world coordinates in meters).

    Returns:
        Dictionary mapping Pepper joint names to angles in radians.
        Suitable for direct use with ALMotion.setAngles().

    Raises:
        ValueError: If world_landmarks has an unexpected shape.
    """
    if world_landmarks.shape != (33, 3):
        raise ValueError(
            f"Expected world_landmarks shape (33, 3), got {world_landmarks.shape}"
        )

    # Compute raw angles for each joint.
    raw_angles: dict[str, float] = {
        "HeadYaw":        _head_yaw(world_landmarks),
        "HeadPitch":      _head_pitch(world_landmarks),
        "LShoulderPitch": _shoulder_pitch(world_landmarks, "left"),
        "LShoulderRoll":  _shoulder_roll(world_landmarks, "left"),
        "LElbowYaw":      _elbow_yaw(world_landmarks, "left"),
        "LElbowRoll":     _elbow_roll(world_landmarks, "left"),
        "LWristYaw":      _wrist_yaw(world_landmarks, "left"),
        "RShoulderPitch": _shoulder_pitch(world_landmarks, "right"),
        "RShoulderRoll":  _shoulder_roll(world_landmarks, "right"),
        "RElbowYaw":      _elbow_yaw(world_landmarks, "right"),
        "RElbowRoll":     _elbow_roll(world_landmarks, "right"),
        "RWristYaw":      _wrist_yaw(world_landmarks, "right"),
    }

    # Clamp all angles to Pepper's physical joint limits.
    clamped: dict[str, float] = {}
    for joint_name, angle in raw_angles.items():
        min_limit, max_limit = PEPPER_JOINTS[joint_name]
        clamped[joint_name] = clamp_angle(angle, min_limit, max_limit)

    return clamped
