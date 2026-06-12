"""
Joint angle calculation utilities.

Provides 3D vector math for computing joint angles from landmark
coordinates. These functions underpin both the Pepper joint mapping
and future body language metrics (arm openness, lean angle, etc.).

All functions operate on NumPy arrays for performance and consistency
with the pipeline's data format.
"""

import numpy as np


def vector_between(point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
    """Compute the 3D vector from point A to point B.

    Args:
        point_a: Source point as (x, y, z) array.
        point_b: Target point as (x, y, z) array.

    Returns:
        3D direction vector (not normalized).
    """
    return point_b - point_a


def calculate_angle(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray,
) -> float:
    """Calculate the angle at vertex B formed by rays BA and BC.

    Uses the dot-product formula:  angle = arccos( (BA · BC) / (|BA| × |BC|) )

    Args:
        point_a: First endpoint as (x, y, z).
        point_b: Vertex (the point at which the angle is measured).
        point_c: Second endpoint as (x, y, z).

    Returns:
        Angle in radians, in the range [0, π].
    """
    vec_ba = vector_between(point_b, point_a)
    vec_bc = vector_between(point_b, point_c)

    # Compute cosine via dot product, clamp to [-1, 1] to avoid
    # numerical issues with arccos at the boundaries.
    norm_ba = np.linalg.norm(vec_ba)
    norm_bc = np.linalg.norm(vec_bc)

    if norm_ba < 1e-8 or norm_bc < 1e-8:
        # Degenerate case: two landmarks are nearly coincident.
        return 0.0

    cosine = np.dot(vec_ba, vec_bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.arccos(cosine))


def rotation_angle_around_axis(
    vec_from: np.ndarray,
    vec_to: np.ndarray,
    axis: np.ndarray,
) -> float:
    """Compute the signed rotation angle of vec_to relative to vec_from
    around a given axis.

    Projects both vectors onto the plane perpendicular to `axis`, then
    computes the signed angle between the projections using atan2.

    This is used for decomposing 3D rotations into Yaw / Roll components
    aligned with Pepper's joint conventions.

    Args:
        vec_from: Reference direction vector (3D).
        vec_to:   Target direction vector (3D).
        axis:     Rotation axis (3D, will be normalized internally).

    Returns:
        Signed angle in radians, in the range (-π, π].
    """
    # Normalize the rotation axis.
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        return 0.0
    axis = axis / axis_norm

    # Project both vectors onto the plane perpendicular to the axis.
    proj_from = vec_from - np.dot(vec_from, axis) * axis
    proj_to = vec_to - np.dot(vec_to, axis) * axis

    norm_from = np.linalg.norm(proj_from)
    norm_to = np.linalg.norm(proj_to)

    if norm_from < 1e-8 or norm_to < 1e-8:
        # One of the vectors is parallel to the axis — angle is undefined.
        return 0.0

    # Compute angle via atan2 for correct sign and quadrant.
    cosine = np.dot(proj_from, proj_to) / (norm_from * norm_to)
    cosine = np.clip(cosine, -1.0, 1.0)

    cross = np.cross(proj_from, proj_to)
    sine_sign = np.dot(cross, axis)

    return float(np.arctan2(sine_sign / (norm_from * norm_to), cosine))


def clamp_angle(angle: float, min_val: float, max_val: float) -> float:
    """Clamp an angle to the range [min_val, max_val].

    Used to enforce Pepper's physical joint limits.

    Args:
        angle:   The angle in radians to clamp.
        min_val: Lower bound (radians).
        max_val: Upper bound (radians).

    Returns:
        Clamped angle.
    """
    return float(np.clip(angle, min_val, max_val))
