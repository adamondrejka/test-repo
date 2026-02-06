"""Tests for pose conversion utilities."""

import numpy as np
import pytest

from utils.matrix import (
    row_major_to_matrix,
    matrix_to_row_major,
    validate_transform,
    extract_position,
    extract_rotation,
    arkit_to_nerfstudio,
    compute_camera_intrinsics_dict,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    slerp,
    interpolate_transforms,
)


class TestMatrixConversion:
    """Tests for matrix format conversion."""

    def test_row_major_to_matrix(self):
        """Test converting row-major array to 4x4 matrix."""
        # Identity matrix in row-major
        identity = [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ]
        result = row_major_to_matrix(identity)

        assert result.shape == (4, 4)
        np.testing.assert_array_almost_equal(result, np.eye(4))

    def test_matrix_to_row_major(self):
        """Test converting 4x4 matrix to row-major array."""
        matrix = np.eye(4)
        matrix[0, 3] = 1.0  # Translation X
        matrix[1, 3] = 2.0  # Translation Y
        matrix[2, 3] = 3.0  # Translation Z

        result = matrix_to_row_major(matrix)

        assert len(result) == 16
        assert result[3] == 1.0  # X translation
        assert result[7] == 2.0  # Y translation
        assert result[11] == 3.0  # Z translation

    def test_roundtrip_conversion(self):
        """Test that row_major -> matrix -> row_major preserves data."""
        original = [
            0.707, -0.707, 0, 1.5,
            0.707, 0.707, 0, 2.5,
            0, 0, 1, 3.5,
            0, 0, 0, 1,
        ]
        matrix = row_major_to_matrix(original)
        result = matrix_to_row_major(matrix)

        np.testing.assert_array_almost_equal(original, result)


class TestTransformValidation:
    """Tests for transform matrix validation."""

    def test_valid_identity(self):
        """Identity matrix should be valid."""
        assert validate_transform(np.eye(4))

    def test_valid_rotation_translation(self):
        """Valid rotation + translation should be valid."""
        matrix = np.eye(4)
        # 90-degree rotation around Z
        matrix[0, 0] = 0
        matrix[0, 1] = -1
        matrix[1, 0] = 1
        matrix[1, 1] = 0
        # Translation
        matrix[0, 3] = 1.0
        matrix[1, 3] = 2.0
        matrix[2, 3] = 3.0

        assert validate_transform(matrix)

    def test_invalid_nan(self):
        """Matrix with NaN should be invalid."""
        matrix = np.eye(4)
        matrix[0, 0] = np.nan

        assert not validate_transform(matrix)

    def test_invalid_inf(self):
        """Matrix with Inf should be invalid."""
        matrix = np.eye(4)
        matrix[1, 1] = np.inf

        assert not validate_transform(matrix)

    def test_invalid_bottom_row(self):
        """Matrix with wrong bottom row should be invalid."""
        matrix = np.eye(4)
        matrix[3, 0] = 0.1  # Should be 0

        assert not validate_transform(matrix)

    def test_invalid_non_orthogonal(self):
        """Non-orthogonal rotation should be invalid."""
        matrix = np.eye(4)
        matrix[0, 0] = 2.0  # Scaling, not pure rotation

        assert not validate_transform(matrix)


class TestPositionExtraction:
    """Tests for position/translation extraction."""

    def test_extract_position(self):
        """Test extracting position from transform."""
        matrix = np.eye(4)
        matrix[0, 3] = 1.5
        matrix[1, 3] = 2.5
        matrix[2, 3] = 3.5

        pos = extract_position(matrix)

        np.testing.assert_array_almost_equal(pos, [1.5, 2.5, 3.5])

    def test_extract_rotation(self):
        """Test extracting rotation from transform."""
        matrix = np.eye(4)
        # 90-degree rotation around Z
        matrix[0, 0] = 0
        matrix[0, 1] = -1
        matrix[1, 0] = 1
        matrix[1, 1] = 0

        rot = extract_rotation(matrix)

        assert rot.shape == (3, 3)
        np.testing.assert_array_almost_equal(
            rot,
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        )


class TestCoordinateConversion:
    """Tests for ARKit to Nerfstudio coordinate conversion."""

    def test_arkit_to_nerfstudio_identity(self):
        """Identity should remain identity (same convention)."""
        identity = np.eye(4)
        result = arkit_to_nerfstudio(identity)

        np.testing.assert_array_almost_equal(result, identity)

    def test_arkit_to_nerfstudio_preserves_transform(self):
        """ARKit and Nerfstudio use same convention, transform preserved."""
        matrix = np.eye(4)
        matrix[0, 3] = 1.0
        matrix[1, 3] = 2.0
        matrix[2, 3] = 3.0

        result = arkit_to_nerfstudio(matrix)

        # Position should be identical
        np.testing.assert_array_almost_equal(
            extract_position(result),
            [1.0, 2.0, 3.0]
        )


class TestIntrinsicsConversion:
    """Tests for camera intrinsics conversion."""

    def test_compute_intrinsics_dict(self):
        """Test converting intrinsic matrix to dict."""
        # Typical iPhone intrinsics (3x3 matrix as 9-element array)
        intrinsics = [
            1500.0, 0, 960.0,
            0, 1500.0, 540.0,
            0, 0, 1,
        ]

        result = compute_camera_intrinsics_dict(intrinsics, 1920, 1080)

        assert result['fl_x'] == 1500.0
        assert result['fl_y'] == 1500.0
        assert result['cx'] == 960.0
        assert result['cy'] == 540.0
        assert result['w'] == 1920
        assert result['h'] == 1080

    def test_invalid_intrinsics_length(self):
        """Invalid intrinsics length should raise error."""
        intrinsics = [1, 2, 3, 4, 5]  # Wrong length

        with pytest.raises(ValueError, match="9 elements"):
            compute_camera_intrinsics_dict(intrinsics, 1920, 1080)


class TestQuaternionConversion:
    """Tests for quaternion utilities."""

    def test_identity_rotation_to_quaternion(self):
        """Identity rotation should give identity quaternion [1, 0, 0, 0]."""
        R = np.eye(3)
        q = rotation_matrix_to_quaternion(R)

        # Should be close to [1, 0, 0, 0] or [-1, 0, 0, 0]
        assert abs(abs(q[0]) - 1.0) < 1e-6
        assert abs(q[1]) < 1e-6
        assert abs(q[2]) < 1e-6
        assert abs(q[3]) < 1e-6

    def test_quaternion_roundtrip(self):
        """quaternion -> matrix -> quaternion should preserve rotation."""
        # 90-degree rotation around Z
        R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=float)

        q = rotation_matrix_to_quaternion(R)
        R_back = quaternion_to_rotation_matrix(q)

        np.testing.assert_array_almost_equal(R, R_back)

    def test_slerp_endpoints(self):
        """SLERP at t=0 and t=1 should return inputs."""
        q1 = np.array([1, 0, 0, 0], dtype=float)
        q2 = np.array([0.707, 0.707, 0, 0], dtype=float)
        q2 = q2 / np.linalg.norm(q2)

        result_0 = slerp(q1, q2, 0.0)
        result_1 = slerp(q1, q2, 1.0)

        np.testing.assert_array_almost_equal(result_0, q1)
        np.testing.assert_array_almost_equal(result_1, q2)


class TestTransformInterpolation:
    """Tests for transform interpolation."""

    def test_interpolate_endpoints(self):
        """Interpolation at t=0 and t=1 should return inputs."""
        t1 = np.eye(4)
        t1[0, 3] = 0.0

        t2 = np.eye(4)
        t2[0, 3] = 10.0

        result_0 = interpolate_transforms(t1, t2, 0.0)
        result_1 = interpolate_transforms(t1, t2, 1.0)

        np.testing.assert_array_almost_equal(
            extract_position(result_0),
            extract_position(t1)
        )
        np.testing.assert_array_almost_equal(
            extract_position(result_1),
            extract_position(t2)
        )

    def test_interpolate_midpoint(self):
        """Interpolation at t=0.5 should be midpoint for translation."""
        t1 = np.eye(4)
        t1[0, 3] = 0.0

        t2 = np.eye(4)
        t2[0, 3] = 10.0

        result = interpolate_transforms(t1, t2, 0.5)

        pos = extract_position(result)
        assert abs(pos[0] - 5.0) < 1e-6  # Should be halfway
