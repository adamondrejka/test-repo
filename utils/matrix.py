"""Matrix transformation utilities for coordinate system conversion."""

import numpy as np
from typing import Tuple, List, Optional


def row_major_to_matrix(data: List[float]) -> np.ndarray:
    """Convert row-major 16-element list to 4x4 matrix."""
    if len(data) != 16:
        raise ValueError(f"Expected 16 elements, got {len(data)}")
    return np.array(data).reshape(4, 4)


def matrix_to_row_major(matrix: np.ndarray) -> List[float]:
    """Convert 4x4 matrix to row-major 16-element list."""
    return matrix.flatten().tolist()


def validate_transform(matrix: np.ndarray) -> bool:
    """
    Validate that a 4x4 matrix is a valid transformation matrix.

    Checks:
    - Shape is (4, 4)
    - No NaN or Inf values
    - Bottom row is [0, 0, 0, 1]
    - Rotation part is orthonormal (within tolerance)
    """
    if matrix.shape != (4, 4):
        return False

    if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
        return False

    # Check bottom row
    if not np.allclose(matrix[3, :], [0, 0, 0, 1], atol=1e-6):
        return False

    # Check rotation is orthonormal
    rotation = matrix[:3, :3]
    should_be_identity = rotation @ rotation.T
    if not np.allclose(should_be_identity, np.eye(3), atol=1e-4):
        return False

    # Check determinant is 1 (not -1, which would indicate reflection)
    det = np.linalg.det(rotation)
    if not np.isclose(det, 1.0, atol=1e-4):
        return False

    return True


def extract_position(matrix: np.ndarray) -> np.ndarray:
    """Extract translation/position from 4x4 transform matrix."""
    return matrix[:3, 3].copy()


def extract_rotation(matrix: np.ndarray) -> np.ndarray:
    """Extract 3x3 rotation matrix from 4x4 transform matrix."""
    return matrix[:3, :3].copy()


def arkit_to_nerfstudio(matrix: np.ndarray) -> np.ndarray:
    """
    Convert ARKit camera transform to Nerfstudio convention.

    ARKit uses:
    - Right-handed coordinate system
    - Y-up
    - Camera looks along -Z axis

    Nerfstudio (OpenGL convention) uses:
    - Right-handed coordinate system
    - Y-up
    - Camera looks along -Z axis

    Both use the same convention, so no conversion needed!
    The transform represents camera-to-world.
    """
    # Both ARKit and Nerfstudio use OpenGL convention
    # No coordinate system flip required
    return matrix.copy()


def compute_camera_intrinsics_dict(
    intrinsic_matrix: List[float],
    width: int,
    height: int
) -> dict:
    """
    Convert ARKit intrinsic matrix to Nerfstudio format.

    ARKit provides a 3x3 intrinsic matrix:
    [fx,  0, cx]
    [ 0, fy, cy]
    [ 0,  0,  1]

    Nerfstudio expects: fl_x, fl_y, cx, cy, w, h
    """
    # Intrinsic matrix is stored row-major as 9 elements
    if len(intrinsic_matrix) != 9:
        raise ValueError(f"Expected 9 elements in intrinsic matrix, got {len(intrinsic_matrix)}")

    K = np.array(intrinsic_matrix).reshape(3, 3)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    return {
        "fl_x": float(fx),
        "fl_y": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "w": width,
        "h": height,
    }


def compute_camera_center(transform: np.ndarray) -> np.ndarray:
    """
    Compute camera center in world coordinates.

    For a camera-to-world transform, the camera center is simply
    the translation component.
    """
    return extract_position(transform)


def compute_view_direction(transform: np.ndarray) -> np.ndarray:
    """
    Compute view direction (where camera is looking) in world coordinates.

    Camera looks along -Z axis in camera space.
    """
    rotation = extract_rotation(transform)
    # Camera looks along -Z in camera space
    view_dir = rotation @ np.array([0, 0, -1])
    return view_dir / np.linalg.norm(view_dir)


def interpolate_transforms(
    t1: np.ndarray,
    t2: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Interpolate between two transforms.

    Uses linear interpolation for translation and
    spherical linear interpolation for rotation.

    Args:
        t1: First transform matrix (4x4)
        t2: Second transform matrix (4x4)
        alpha: Interpolation factor (0=t1, 1=t2)

    Returns:
        Interpolated transform matrix
    """
    # Interpolate position linearly
    pos1 = extract_position(t1)
    pos2 = extract_position(t2)
    pos_interp = pos1 * (1 - alpha) + pos2 * alpha

    # Interpolate rotation using quaternions (SLERP)
    rot1 = extract_rotation(t1)
    rot2 = extract_rotation(t2)

    # Convert to quaternions
    q1 = rotation_matrix_to_quaternion(rot1)
    q2 = rotation_matrix_to_quaternion(rot2)

    # SLERP
    q_interp = slerp(q1, q2, alpha)

    # Convert back to rotation matrix
    rot_interp = quaternion_to_rotation_matrix(q_interp)

    # Compose result
    result = np.eye(4)
    result[:3, :3] = rot_interp
    result[:3, 3] = pos_interp

    return result


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q

    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

    return R


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions."""
    # Normalize
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Compute dot product
    dot = np.dot(q1, q2)

    # If negative dot, negate one quaternion to take shorter path
    if dot < 0:
        q2 = -q2
        dot = -dot

    # If nearly parallel, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    # SLERP
    theta_0 = np.arccos(dot)
    theta = theta_0 * t

    q_perp = q2 - q1 * dot
    q_perp = q_perp / np.linalg.norm(q_perp)

    return q1 * np.cos(theta) + q_perp * np.sin(theta)
