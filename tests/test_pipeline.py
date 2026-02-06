"""Integration tests for the processing pipeline."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def create_test_manifest(output_dir: Path, num_poses: int = 100) -> Path:
    """Create a minimal test scan manifest."""
    # Generate test poses along a path
    poses = []
    for i in range(num_poses):
        t = i / num_poses
        # Simple path: move forward with slight curve
        x = t * 5.0
        y = 1.6  # Eye height
        z = np.sin(t * np.pi) * 2.0

        # Identity rotation with translation
        transform = [
            1, 0, 0, x,
            0, 1, 0, y,
            0, 0, 1, z,
            0, 0, 0, 1,
        ]

        poses.append({
            "timestamp": i * 0.033,  # ~30fps
            "transform_matrix": transform,
            "tracking_state": "normal",
        })

    manifest = {
        "scan_id": "test-scan-001",
        "timestamp": "2024-01-15T10:30:00Z",
        "scan_type": "INDOOR_HYBRID",
        "calibration": {
            "intrinsic_matrix": [
                1500.0, 0, 960.0,
                0, 1500.0, 540.0,
                0, 0, 1,
            ],
            "image_width": 1920,
            "image_height": 1080,
        },
        "assets": {
            "video_path": "video.mov",
            "room_geometry": None,
        },
        "poses": poses,
        "floorplan": {
            "walls": [
                {"start": [0, 0], "end": [5, 0], "height": 2.5},
                {"start": [5, 0], "end": [5, 4], "height": 2.5},
                {"start": [5, 4], "end": [0, 4], "height": 2.5},
                {"start": [0, 4], "end": [0, 0], "height": 2.5},
            ],
            "doors": [
                {"position": [2.5, 0], "width": 0.9, "height": 2.1, "type": "single"},
            ],
            "windows": [
                {"position": [5, 2], "width": 1.2, "height": 1.0, "sill_height": 0.9},
            ],
        },
    }

    manifest_path = output_dir / "scan_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


class TestValidation:
    """Tests for validation utilities."""

    def test_validate_valid_manifest(self, tmp_path):
        """Valid manifest should pass validation."""
        from utils.validation import validate_manifest

        manifest_path = create_test_manifest(tmp_path)
        is_valid, manifest, errors = validate_manifest(manifest_path)

        assert is_valid
        assert manifest is not None
        assert len(errors) == 0
        assert manifest.scan_id == "test-scan-001"

    def test_validate_missing_manifest(self, tmp_path):
        """Missing manifest should fail validation."""
        from utils.validation import validate_manifest

        is_valid, manifest, errors = validate_manifest(tmp_path / "nonexistent.json")

        assert not is_valid
        assert manifest is None
        assert len(errors) > 0

    def test_validate_insufficient_poses(self, tmp_path):
        """Manifest with too few poses should fail."""
        from utils.validation import validate_manifest

        # Create manifest with only 5 poses
        manifest_path = create_test_manifest(tmp_path, num_poses=5)

        is_valid, manifest, errors = validate_manifest(manifest_path)

        assert not is_valid
        assert "10 poses required" in str(errors)

    def test_validate_poses_quality(self, tmp_path):
        """Test pose sequence validation."""
        from utils.validation import validate_poses, PoseEntry

        manifest_path = create_test_manifest(tmp_path, num_poses=50)
        with open(manifest_path) as f:
            data = json.load(f)

        poses = [PoseEntry(**p) for p in data["poses"]]
        is_valid, stats, warnings = validate_poses(poses)

        assert is_valid
        assert stats["total_poses"] == 50
        assert stats["normal_tracking"] == 50
        assert stats["duration"] > 0


class TestPoseConversion:
    """Tests for pose format conversion."""

    def test_create_transforms_json(self, tmp_path):
        """Test creating Nerfstudio transforms.json."""
        from pipeline.convert_poses import create_transforms_json

        # Create test frames
        frames = [
            {
                "image_path": f"./frames/frame_{i:06d}.jpg",
                "transform_matrix": [
                    1, 0, 0, float(i),
                    0, 1, 0, 1.6,
                    0, 0, 1, 0,
                    0, 0, 0, 1,
                ],
                "timestamp": i * 0.033,
            }
            for i in range(20)
        ]

        calibration = {
            "intrinsic_matrix": [
                1500.0, 0, 960.0,
                0, 1500.0, 540.0,
                0, 0, 1,
            ],
            "image_width": 1920,
            "image_height": 1080,
        }

        output_path = tmp_path / "transforms.json"
        result = create_transforms_json(frames, calibration, output_path)

        assert output_path.exists()
        assert result["fl_x"] == 1500.0
        assert result["w"] == 1920
        assert len(result["frames"]) == 20

    def test_convert_from_manifest(self, tmp_path):
        """Test converting full manifest to transforms."""
        from pipeline.convert_poses import convert_from_manifest

        # Create test manifest
        manifest_path = create_test_manifest(tmp_path)

        # Create fake frames directory
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(100):
            (frames_dir / f"frame_{i:06d}.jpg").touch()

        output_path = tmp_path / "transforms.json"
        result = convert_from_manifest(manifest_path, frames_dir, output_path)

        assert output_path.exists()
        assert len(result["frames"]) == 100


class TestCollision:
    """Tests for collision mesh generation."""

    def test_generate_floorplan_svg(self, tmp_path):
        """Test SVG floorplan generation."""
        from pipeline.collision import generate_floorplan_svg

        floorplan_data = {
            "walls": [
                {"start": [0, 0], "end": [5, 0], "height": 2.5},
                {"start": [5, 0], "end": [5, 4], "height": 2.5},
                {"start": [5, 4], "end": [0, 4], "height": 2.5},
                {"start": [0, 4], "end": [0, 0], "height": 2.5},
            ],
            "doors": [
                {"position": [2.5, 0], "width": 0.9, "height": 2.1, "type": "single"},
            ],
            "windows": [],
        }

        output_path = generate_floorplan_svg(
            floorplan_data,
            tmp_path / "floorplan.svg"
        )

        assert output_path.exists()
        content = output_path.read_text()
        assert "<svg" in content
        assert "<line" in content  # Has wall lines

    def test_generate_collision_from_floorplan(self, tmp_path):
        """Test collision mesh from floorplan."""
        from pipeline.collision import generate_collision_from_floorplan

        floorplan_data = {
            "walls": [
                {"start": [0, 0], "end": [5, 0], "height": 2.5},
                {"start": [5, 0], "end": [5, 4], "height": 2.5},
            ],
            "doors": [],
            "windows": [],
        }

        collision_path, stats = generate_collision_from_floorplan(
            floorplan_data,
            tmp_path
        )

        assert collision_path.exists()
        assert stats["wall_count"] == 2
        assert stats["vertices"] > 0


class TestPackaging:
    """Tests for tour package creation."""

    def test_create_metadata(self, tmp_path):
        """Test metadata.json creation."""
        from pipeline.package import create_metadata

        manifest = {
            "scan_id": "test-001",
            "timestamp": "2024-01-15T10:00:00Z",
            "scan_type": "INDOOR_HYBRID",
            "calibration": {"image_width": 1920, "image_height": 1080},
            "poses": [
                {"transform_matrix": [1, 0, 0, 0, 0, 1, 0, 1.6, 0, 0, 1, 0, 0, 0, 0, 1]},
            ],
            "floorplan": None,
        }

        stats = {"stage1": {"duration": 1.0}}

        metadata_path = create_metadata("test-001", manifest, stats, tmp_path / "metadata.json")

        assert metadata_path.exists()
        with open(metadata_path) as f:
            data = json.load(f)

        assert data["scan_id"] == "test-001"
        assert data["viewer"]["camera_height"] == 1.6
        assert "splat" in data["assets"]

    def test_validate_package(self, tmp_path):
        """Test package validation."""
        from pipeline.package import validate_package, create_metadata

        # Create minimal valid package
        (tmp_path / "scene.spz").write_bytes(b"dummy")

        manifest = {"scan_id": "test-001"}
        create_metadata("test-001", manifest, {}, tmp_path / "metadata.json")

        errors = validate_package(tmp_path)

        assert len(errors) == 0

    def test_validate_incomplete_package(self, tmp_path):
        """Test validation catches missing files."""
        from pipeline.package import validate_package

        # Empty directory
        errors = validate_package(tmp_path)

        assert len(errors) > 0
        assert any("scene.spz" in e for e in errors)


class TestCompression:
    """Tests for compression utilities."""

    def test_get_ply_info(self, tmp_path):
        """Test reading PLY file info."""
        from pipeline.compress import get_ply_info

        # Create minimal PLY file
        ply_content = """ply
format ascii 1.0
element vertex 10
property float x
property float y
property float z
end_header
0 0 0
1 0 0
2 0 0
3 0 0
4 0 0
5 0 0
6 0 0
7 0 0
8 0 0
9 0 0
"""
        ply_path = tmp_path / "test.ply"
        ply_path.write_text(ply_content)

        info = get_ply_info(ply_path)

        assert info["point_count"] == 10
        assert info["property_count"] == 3


class TestMatrixUtils:
    """Additional matrix utility tests."""

    def test_velocity_calculation(self):
        """Test camera velocity calculation."""
        from utils.matrix import row_major_to_matrix

        # Two transforms 1 second apart, 2 meters apart
        t1 = row_major_to_matrix([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ])
        t2 = row_major_to_matrix([
            1, 0, 0, 2,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ])

        pos1 = t1[:3, 3]
        pos2 = t2[:3, 3]
        distance = np.linalg.norm(pos2 - pos1)
        velocity = distance / 1.0  # 1 second

        assert abs(velocity - 2.0) < 1e-6
