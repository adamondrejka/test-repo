"""Validation utilities for scan packages and data integrity."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator
import numpy as np

from .matrix import row_major_to_matrix, validate_transform


# Pydantic models for scan manifest validation


class CameraCalibration(BaseModel):
    intrinsic_matrix: List[float] = Field(..., min_length=9, max_length=9)
    image_width: int = Field(..., gt=0)
    image_height: int = Field(..., gt=0)

    @field_validator("intrinsic_matrix")
    @classmethod
    def validate_intrinsics(cls, v):
        if len(v) != 9:
            raise ValueError("Intrinsic matrix must have exactly 9 elements")
        # Check for valid focal lengths (positive)
        K = np.array(v).reshape(3, 3)
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("Focal lengths must be positive")
        return v


class AssetPaths(BaseModel):
    video_path: Optional[str] = None
    images_dir: Optional[str] = None
    room_geometry: Optional[str] = None
    mesh_path: Optional[str] = None


class PoseEntry(BaseModel):
    timestamp: float = Field(..., ge=0)
    transform_matrix: List[float] = Field(..., min_length=16, max_length=16)
    tracking_state: str = Field(default="normal")
    image_name: Optional[str] = None

    @field_validator("transform_matrix")
    @classmethod
    def validate_transform_matrix(cls, v):
        if len(v) != 16:
            raise ValueError("Transform matrix must have exactly 16 elements")
        matrix = row_major_to_matrix(v)
        if not validate_transform(matrix):
            raise ValueError("Invalid transformation matrix")
        return v

    @field_validator("tracking_state")
    @classmethod
    def validate_tracking_state(cls, v):
        valid_states = {"normal", "limited", "not_available"}
        if v not in valid_states:
            raise ValueError(f"Invalid tracking state: {v}. Must be one of {valid_states}")
        return v


class WallSegment(BaseModel):
    start: List[float] = Field(..., min_length=2, max_length=2)
    end: List[float] = Field(..., min_length=2, max_length=2)
    height: float = Field(..., gt=0)


class DoorEntry(BaseModel):
    position: List[float] = Field(..., min_length=2, max_length=2)
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)
    type: str = Field(default="single")


class WindowEntry(BaseModel):
    position: List[float] = Field(..., min_length=2, max_length=2)
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)
    sill_height: float = Field(default=0, ge=0)


class FloorplanData(BaseModel):
    walls: List[WallSegment] = Field(default_factory=list)
    doors: List[DoorEntry] = Field(default_factory=list)
    windows: List[WindowEntry] = Field(default_factory=list)


class ScanManifest(BaseModel):
    """Pydantic model for scan manifest validation."""

    scan_id: str = Field(alias="scan_id")
    timestamp: str
    scan_type: str = Field(alias="scan_type")
    calibration: CameraCalibration
    assets: AssetPaths
    poses: List[PoseEntry]
    floorplan: Optional[FloorplanData] = None
    video_start_time: Optional[float] = Field(default=None, alias="video_start_time")

    model_config = {"populate_by_name": True}

    @field_validator("poses")
    @classmethod
    def validate_poses_list(cls, v):
        if len(v) < 10:
            raise ValueError(f"At least 10 poses required, got {len(v)}")
        return v


def validate_manifest(manifest_path: Path) -> Tuple[bool, Optional[ScanManifest], List[str]]:
    """
    Validate a scan manifest JSON file.

    Args:
        manifest_path: Path to scan_manifest.json

    Returns:
        Tuple of (is_valid, parsed_manifest, list_of_errors)
    """
    errors = []

    if not manifest_path.exists():
        return False, None, ["Manifest file does not exist"]

    try:
        with open(manifest_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, None, [f"Invalid JSON: {e}"]

    try:
        manifest = ScanManifest(**data)
        return True, manifest, []
    except Exception as e:
        errors.append(str(e))
        return False, None, errors


def validate_video_file(video_path: Path) -> Tuple[bool, Dict, List[str]]:
    """
    Validate video file exists and has expected properties.

    Returns:
        Tuple of (is_valid, video_info, list_of_errors)
    """
    errors = []
    info = {}

    if not video_path.exists():
        return False, info, ["Video file does not exist"]

    # Get file size
    file_size = video_path.stat().st_size
    info["file_size"] = file_size

    if file_size < 1024:  # Less than 1KB
        errors.append("Video file too small")

    # Check extension
    if video_path.suffix.lower() not in [".mov", ".mp4", ".m4v"]:
        errors.append(f"Unexpected video format: {video_path.suffix}")

    # Try to probe video with ffprobe (if available)
    try:
        import subprocess

        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(video_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            probe_data = json.loads(result.stdout)
            video_stream = next(
                (s for s in probe_data.get("streams", []) if s["codec_type"] == "video"), None
            )
            if video_stream:
                info["width"] = int(video_stream.get("width", 0))
                info["height"] = int(video_stream.get("height", 0))
                info["duration"] = float(probe_data.get("format", {}).get("duration", 0))
                info["codec"] = video_stream.get("codec_name", "unknown")

                # Get frame rate
                fps_str = video_stream.get("r_frame_rate", "0/1")
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    info["fps"] = float(num) / float(den) if float(den) != 0 else 0
    except (FileNotFoundError, json.JSONDecodeError):
        # ffprobe not available, skip detailed validation
        pass

    return len(errors) == 0, info, errors


def validate_poses(poses: List[PoseEntry]) -> Tuple[bool, Dict, List[str]]:
    """
    Validate pose sequence for quality and consistency.

    Checks:
    - Timestamps are monotonically increasing
    - No large gaps in timestamps
    - Sufficient pose density
    - No sudden position jumps

    Returns:
        Tuple of (is_valid, stats, list_of_warnings)
    """
    warnings = []
    stats = {
        "total_poses": len(poses),
        "normal_tracking": 0,
        "limited_tracking": 0,
        "lost_tracking": 0,
        "duration": 0,
        "avg_fps": 0,
    }

    if len(poses) < 10:
        return False, stats, ["Insufficient poses (minimum 10 required)"]

    # Track timestamps
    timestamps = [p.timestamp for p in poses]
    stats["duration"] = timestamps[-1] - timestamps[0]

    if stats["duration"] > 0:
        stats["avg_fps"] = len(poses) / stats["duration"]

    # Check monotonicity
    for i in range(1, len(timestamps)):
        if timestamps[i] <= timestamps[i - 1]:
            warnings.append(f"Non-monotonic timestamp at index {i}")

    # Check for gaps
    max_gap = 0.5  # 500ms
    for i in range(1, len(timestamps)):
        gap = timestamps[i] - timestamps[i - 1]
        if gap > max_gap:
            warnings.append(f"Large timestamp gap ({gap:.2f}s) at index {i}")

    # Count tracking states
    for pose in poses:
        if pose.tracking_state == "normal":
            stats["normal_tracking"] += 1
        elif pose.tracking_state == "limited":
            stats["limited_tracking"] += 1
        else:
            stats["lost_tracking"] += 1

    # Check tracking quality
    normal_ratio = stats["normal_tracking"] / len(poses)
    if normal_ratio < 0.5:
        warnings.append(f"Low tracking quality: only {normal_ratio * 100:.1f}% normal tracking")

    # Check for position jumps
    prev_pos = None
    max_velocity = 5.0  # m/s - reasonable walking speed
    for i, pose in enumerate(poses):
        matrix = row_major_to_matrix(pose.transform_matrix)
        pos = matrix[:3, 3]

        if prev_pos is not None and i > 0:
            dt = timestamps[i] - timestamps[i - 1]
            if dt > 0:
                velocity = np.linalg.norm(pos - prev_pos) / dt
                if velocity > max_velocity:
                    warnings.append(f"High velocity ({velocity:.1f} m/s) at index {i}")

        prev_pos = pos

    is_valid = len([w for w in warnings if "Low tracking" not in w]) == 0
    return is_valid, stats, warnings


def validate_scan_package(package_dir: Path) -> Tuple[bool, Dict, List[str]]:
    """
    Validate a complete scan package directory.

    Expected structure:
    - scan_manifest.json
    - video.mov (or video_path from manifest)
    - room.usdz (optional)

    Returns:
        Tuple of (is_valid, package_info, list_of_errors)
    """
    errors = []
    info = {}

    # Check manifest
    manifest_path = package_dir / "scan_manifest.json"
    manifest_valid, manifest, manifest_errors = validate_manifest(manifest_path)

    if not manifest_valid:
        return False, info, manifest_errors

    info["manifest"] = manifest

    # Check for images directory or video (at least one required)
    has_images = False
    has_video = False

    if manifest.assets.images_dir:
        images_dir = package_dir / manifest.assets.images_dir
        if images_dir.exists() and images_dir.is_dir():
            image_files = list(images_dir.glob("frame_*.jpg"))
            if len(image_files) > 0:
                has_images = True
                info["images"] = {"count": len(image_files), "dir": str(images_dir)}
            else:
                errors.append(f"Images directory exists but contains no frame_*.jpg files")

    if manifest.assets.video_path:
        video_path = package_dir / manifest.assets.video_path
        video_valid, video_info, video_errors = validate_video_file(video_path)
        if video_valid:
            has_video = True
        errors.extend(video_errors)
        info["video"] = video_info

    if not has_images and not has_video:
        errors.append("Scan package must contain either an images directory or a video file")

    # Check room geometry (optional - just a warning, not an error)
    if manifest.assets.room_geometry:
        room_path = package_dir / manifest.assets.room_geometry
        if room_path.exists():
            info["has_room_geometry"] = True
        # Missing room geometry is not a validation error - it's optional

    # Validate poses
    poses_valid, poses_stats, poses_warnings = validate_poses(manifest.poses)
    info["poses"] = poses_stats

    # Add warnings as errors if critical
    for warning in poses_warnings:
        if "Low tracking" in warning or "Insufficient" in warning:
            errors.append(warning)

    is_valid = len(errors) == 0
    return is_valid, info, errors
