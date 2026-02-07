#!/usr/bin/env python3
"""
Diagnose Scan Tool
==================

Verifies iOS scan data integrity WITHOUT any coordinate conversions.
Answers: "Are the raw iOS data geometrically consistent?"

Checks:
- Frame dimensions vs calibration dimensions
- Principal point position relative to frame center
- Camera height (Y component, should be ~1-2m)
- Camera path length and velocity
- Rotation matrix validity (orthonormal, det=1)
- Projects raw ARKit world axes onto frames

Usage:
    python backend/scripts/diagnose_scan.py <path_to_zip_or_dir> --out diagnosis
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.ingest import ingest
from utils.matrix import row_major_to_matrix, validate_transform


def extract_frame(video_path: Path, timestamp: float, output_path: Path):
    """Extract a single frame from video at timestamp, no auto-rotation."""
    cmd = [
        'ffmpeg', '-y', '-noautorotate',
        '-ss', str(timestamp), '-i', str(video_path),
        '-frames:v', '1', '-q:v', '2', str(output_path), '-v', 'quiet'
    ]
    subprocess.run(cmd, check=True)


def draw_axis_raw(img, K, R_wc, t_wc, axis_length=0.5):
    """Draw world coordinate axes using raw ARKit poses (with Y/Z flip for OpenCV)."""
    points_3d = np.float32([
        [0, 0, 0],
        [axis_length, 0, 0],  # X - Red
        [0, axis_length, 0],  # Y - Green (ARKit up)
        [0, 0, axis_length],  # Z - Blue
    ])

    points_2d, _ = cv2.projectPoints(points_3d, R_wc, t_wc, K, None)

    pts = points_2d.reshape(-1, 2).astype(np.int32).tolist()
    origin = tuple(pts[0])
    x_end = tuple(pts[1])
    y_end = tuple(pts[2])
    z_end = tuple(pts[3])

    cv2.line(img, origin, x_end, (0, 0, 255), 4)   # X - Red
    cv2.line(img, origin, y_end, (0, 255, 0), 4)    # Y - Green
    cv2.line(img, origin, z_end, (255, 0, 0), 4)    # Z - Blue

    # Label axes
    cv2.putText(img, "X", x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Y", y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, "Z", z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return img


def draw_floor_grid(img, K, R_wc, t_wc, floor_y=0.0, size=3.0, steps=10):
    """Draw a floor grid on the XZ plane at given Y height."""
    half = size / 2
    step = size / steps

    points = []
    # Lines along Z (constant X)
    for i in range(steps + 1):
        x = -half + i * step
        points.append([x, floor_y, -half])
        points.append([x, floor_y, half])
    # Lines along X (constant Z)
    for i in range(steps + 1):
        z = -half + i * step
        points.append([-half, floor_y, z])
        points.append([half, floor_y, z])

    pts_3d = np.float32(points)
    pts_2d, _ = cv2.projectPoints(pts_3d, R_wc, t_wc, K, None)
    pts_2d = pts_2d.reshape(-1, 2).astype(int)

    h, w = img.shape[:2]
    for i in range(0, len(pts_2d), 2):
        p1 = tuple(int(v) for v in pts_2d[i])
        p2 = tuple(int(v) for v in pts_2d[i + 1])
        if (-w < p1[0] < 2 * w and -h < p1[1] < 2 * h and
                -w < p2[0] < 2 * w and -h < p2[1] < 2 * h):
            cv2.line(img, p1, p2, (180, 180, 180), 1)

    return img


def diagnose_scan(package_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(package_dir / "scan_manifest.json") as f:
        manifest = json.load(f)

    video_path = package_dir / manifest['assets']['video_path']
    poses = manifest['poses']
    calibration = manifest['calibration']
    video_start_time = manifest.get('video_start_time', poses[0]['timestamp'])
    capture_orientation = manifest.get('capture_orientation', 'unknown')

    # Camera intrinsics (raw from ARKit, row-major 3x3)
    K_raw = np.array(calibration['intrinsic_matrix']).reshape(3, 3)
    cal_w = calibration['image_width']
    cal_h = calibration['image_height']
    fx, fy = K_raw[0, 0], K_raw[1, 1]
    cx, cy = K_raw[0, 2], K_raw[1, 2]

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # ---- Diagnostic Report ----
    report = []
    report.append("=" * 60)
    report.append("SCAN DIAGNOSTIC REPORT")
    report.append("=" * 60)
    report.append(f"Capture orientation: {capture_orientation}")
    report.append(f"Calibration dimensions: {cal_w} x {cal_h}")
    report.append(f"Intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    report.append(f"Principal point offset from center: "
                  f"dx={cx - cal_w / 2:.1f}px  dy={cy - cal_h / 2:.1f}px")
    report.append(f"Poses: {len(poses)}")
    report.append(f"Video start time: {video_start_time:.3f}")
    report.append("")

    # Validate all poses
    positions = []
    invalid_count = 0
    for i, pose in enumerate(poses):
        mat = row_major_to_matrix(pose['transform_matrix'])
        if not validate_transform(mat):
            invalid_count += 1
            continue
        positions.append(mat[:3, 3])

    positions = np.array(positions)
    report.append(f"Valid poses: {len(positions)}/{len(poses)}")
    report.append(f"Invalid poses: {invalid_count}")

    # Camera heights (Y component in ARKit = up)
    heights = positions[:, 1]
    report.append(f"Camera height (Y): min={heights.min():.2f}m  "
                  f"max={heights.max():.2f}m  mean={heights.mean():.2f}m")

    # Path length
    deltas = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    path_length = segment_lengths.sum()
    report.append(f"Camera path length: {path_length:.2f}m")

    # Velocity stats
    timestamps = np.array([p['timestamp'] for p in poses if validate_transform(
        row_major_to_matrix(p['transform_matrix']))])
    dt = np.diff(timestamps)
    velocities = segment_lengths / np.maximum(dt, 1e-6)
    report.append(f"Velocity: mean={velocities.mean():.2f}m/s  "
                  f"max={velocities.max():.2f}m/s")

    # Estimate floor height (camera Y minus ~1.5m)
    floor_y = heights.mean() - 1.5
    report.append(f"Estimated floor Y: {floor_y:.2f}m")
    report.append("")

    # ---- Extract and annotate sample frames ----
    num_samples = min(5, len(poses))
    indices = np.linspace(0, len(poses) - 1, num_samples, dtype=int)

    for idx in indices:
        pose = poses[idx]
        video_time = pose['timestamp'] - video_start_time
        if video_time < 0:
            continue

        frame_path = output_dir / f"diag_{idx:04d}.jpg"
        try:
            extract_frame(video_path, video_time, frame_path)
        except subprocess.CalledProcessError:
            report.append(f"Frame {idx}: FAILED to extract at t={video_time:.2f}s")
            continue

        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        frame_h, frame_w = img.shape[:2]

        # Check frame vs calibration
        dim_match = (frame_w == cal_w and frame_h == cal_h)
        dim_status = "MATCH" if dim_match else "MISMATCH"
        report.append(f"Frame {idx}: {frame_w}x{frame_h} vs cal {cal_w}x{cal_h} -> {dim_status}")

        # Get raw ARKit C2W pose
        T_c2w = row_major_to_matrix(pose['transform_matrix'])

        # Invert to get W2C
        R_c2w = T_c2w[:3, :3]
        t_c2w = T_c2w[:3, 3]

        # ARKit camera: Y-up, looks along -Z
        # OpenCV camera: Y-down, looks along +Z
        # To project correctly with cv2.projectPoints, we need to flip Y and Z
        # of the world-to-camera transform
        flip = np.diag([1.0, -1.0, -1.0])
        R_wc_cv = flip @ R_c2w.T
        t_wc_cv = flip @ (-R_c2w.T @ t_c2w)

        # Draw floor grid and axes
        img = draw_floor_grid(img, K, R_wc_cv, t_wc_cv, floor_y=floor_y)
        img = draw_axis_raw(img, K, R_wc_cv, t_wc_cv)

        # Annotate
        pos = T_c2w[:3, 3]
        cv2.putText(img, f"Frame {idx}  t={video_time:.2f}s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(img, f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img, f"{frame_w}x{frame_h} [{dim_status}]", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if dim_match else (0, 0, 255), 2)

        cv2.imwrite(str(frame_path), img)

    # Print and save report
    report_text = "\n".join(report)
    print(report_text)

    with open(output_dir / "report.txt", 'w') as f:
        f.write(report_text)

    print(f"\nAnnotated frames and report saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose iOS scan data integrity")
    parser.add_argument("input", help="Path to input .zip or directory")
    parser.add_argument("--out", default="diagnosis", help="Output directory")

    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.out)

    if input_path.suffix == '.zip':
        work_dir = Path(tempfile.mkdtemp())
        package_dir, _, _ = ingest(input_path, work_dir, validate=False)
    else:
        package_dir = input_path
        work_dir = None

    try:
        diagnose_scan(package_dir, output_dir)
    finally:
        if work_dir is not None:
            shutil.rmtree(work_dir)
