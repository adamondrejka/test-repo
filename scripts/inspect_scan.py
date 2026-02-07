#!/usr/bin/env python3
"""
Inspect Scan Tool
=================

Diagnostic tool to visualize synchronization and geometric alignment of a scan package.
Projects 3D world axes and a floor grid onto video frames to verify:
1. Temporal sync (does camera movement match video flow?)
2. Geometric alignment (is gravity down? is the floor flat?)
3. Intrinsics (does the FOV match?)

Usage:
    python backend/scripts/inspect_scan.py <path_to_zip_or_dir>
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.ingest import ingest
from utils.matrix import row_major_to_matrix, arkit_to_nerfstudio

def draw_axis(img, R, t, K, dist=None, axis_length=0.3):
    """Draw 3D coordinate axes on an image."""
    # Define local axis points (Origin, X, Y, Z)
    points_3d = np.float32([
        [0, 0, 0],          # Origin
        [axis_length, 0, 0], # X (Red)
        [0, axis_length, 0], # Y (Green - Up)
        [0, 0, axis_length]  # Z (Blue - Forward)
    ])

    # Transform to camera coordinates: P_cam = R * P_world + t
    # Note: R and t here are World-to-Camera (Extrinsics)
    
    # Project 3D points to 2D image plane
    points_2d, _ = cv2.projectPoints(points_3d, R, t, K, dist)
    
    origin = tuple(points_2d[0].ravel().astype(int))
    x_axis = tuple(points_2d[1].ravel().astype(int))
    y_axis = tuple(points_2d[2].ravel().astype(int))
    z_axis = tuple(points_2d[3].ravel().astype(int))

    # Draw lines (BGR format for OpenCV)
    img = cv2.line(img, origin, x_axis, (0, 0, 255), 5)  # X - Red
    img = cv2.line(img, origin, y_axis, (0, 255, 0), 5)  # Y - Green
    img = cv2.line(img, origin, z_axis, (255, 0, 0), 5)  # Z - Blue
    
    return img

def draw_grid(img, R, t, K, floor_y=0.0, size=2.0, steps=10):
    """Draw a 3D floor grid (XZ plane) at a specific Y height."""
    half_size = size / 2
    step_size = size / steps
    
    lines = []
    
    # Z-lines (constant X)
    for i in range(steps + 1):
        x = -half_size + i * step_size
        lines.append([
            [x, floor_y, -half_size],
            [x, floor_y, half_size]
        ])
        
    # X-lines (constant Z)
    for i in range(steps + 1):
        z = -half_size + i * step_size
        lines.append([
            [-half_size, floor_y, z],
            [half_size, floor_y, z]
        ])
    
    # Flatten points
    all_points = []
    for line in lines:
        all_points.extend(line)
        
    points_3d = np.float32(all_points)
    
    # Project
    points_2d, _ = cv2.projectPoints(points_3d, R, t, K, None)
    points_2d = points_2d.reshape(-1, 2).astype(int)
    
    # Draw lines
    for i in range(0, len(points_2d), 2):
        pt1 = tuple(points_2d[i])
        pt2 = tuple(points_2d[i+1])
        
        # Check bounds roughly (don't draw lines crossing behind camera)
        if (0 <= pt1[0] < img.shape[1] * 2 and 0 <= pt1[1] < img.shape[0] * 2 and
            0 <= pt2[0] < img.shape[1] * 2 and 0 <= pt2[1] < img.shape[0] * 2):
            img = cv2.line(img, pt1, pt2, (200, 200, 200), 1)
            
    return img

def extract_single_frame(video_path, timestamp, output_path, transpose=None):
    """Extract a single frame from video at timestamp using FFmpeg."""
    cmd = [
        'ffmpeg', '-y', '-noautorotate', '-ss', str(timestamp), '-i', str(video_path)
    ]
    
    if transpose is not None:
        cmd.extend(['-vf', f'transpose={transpose}'])
        
    cmd.extend([
        '-frames:v', '1', '-q:v', '2', str(output_path), '-v', 'quiet'
    ])
    subprocess.run(cmd, check=True)

def analyze_scan(package_dir: Path, output_dir: Path, transpose: Optional[int] = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Manifest
    with open(package_dir / "scan_manifest.json") as f:
        manifest = json.load(f)
        
    video_path = package_dir / manifest['assets']['video_path']
    poses = manifest['poses']
    calibration = manifest['calibration']
    
    # Video Start Time (Sync Offset)
    video_start_time = manifest.get('video_start_time', poses[0]['timestamp'])
    print(f"Video Start Time: {video_start_time:.3f}")
    
    # Camera Intrinsics
    intrinsics = np.array(calibration['intrinsic_matrix']).reshape(3, 3)
    w = calibration['image_width']
    h = calibration['image_height']
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Apply transposition to intrinsics if needed
    if transpose == 2: # 90 CCW
        new_fx, new_fy = fy, fx
        new_cx = cy
        new_cy = w - cx
        fx, fy, cx, cy = new_fx, new_fy, new_cx, new_cy
        w, h = h, w
        print(f"Adjusted intrinsics for 90deg CCW")
    elif transpose == 1: # 90 CW
        new_fx, new_fy = fy, fx
        new_cx = h - cy
        new_cy = cx
        fx, fy, cx, cy = new_fx, new_fy, new_cx, new_cy
        w, h = h, w
        print(f"Adjusted intrinsics for 90deg CW")
        
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 1. Plot Velocity / Sync Check
    timestamps = []
    velocities = []
    positions = []
    prev_pos, prev_time = None, None
    
    for pose in poses:
        mat = row_major_to_matrix(pose['transform_matrix'])
        pos = mat[:3, 3]
        time_val = pose['timestamp']
        positions.append(pos)
        timestamps.append(time_val)
        if prev_pos is not None:
            dt = time_val - prev_time
            velocities.append(np.linalg.norm(pos - prev_pos) / dt if dt > 0 else 0)
        else:
            velocities.append(0)
        prev_pos, prev_time = pos, time_val
        
    positions = np.array(positions)
    rel_timestamps = np.array(timestamps) - video_start_time
    
    plt.figure(figsize=(10, 6))
    plt.plot(rel_timestamps, velocities, label='Camera Velocity')
    plt.axvline(x=0, color='r', linestyle='--', label='Video Start')
    plt.xlabel("Time relative to Video Start (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Synchronization Check")
    plt.grid(True)
    plt.savefig(output_dir / "sync_check.png")
    plt.close()
    
    # 2. Visualize Frames
    mean_y = np.mean(positions[:, 1])
    floor_y = mean_y - 1.5
    
    num_samples = 6
    indices = np.linspace(0, len(poses)-1, num_samples, dtype=int)
    
    for idx in tqdm(indices, desc="Generating checks"):
        pose = poses[idx]
        video_time = pose['timestamp'] - video_start_time
        if video_time < 0: continue
            
        frame_path = output_dir / f"check_{idx:04d}.jpg"
        extract_single_frame(video_path, video_time, frame_path, transpose=transpose)
        
        img = cv2.imread(str(frame_path))
        if img is None: continue
            
        # Transform Pose
        T_cw = row_major_to_matrix(pose['transform_matrix'])
        
        # Convert ARKit -> Nerfstudio convention (Y-down, +Z forward)
        # This matches how we train, so our debug grid will match training view
        T_ns = arkit_to_nerfstudio(T_cw)
        
        # Apply transpose rotation to camera matrix
        if transpose == 2: # 90 CCW in image -> 90 CW around Z in camera
            R_z_cw = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            T_ns = T_ns @ R_z_cw
        elif transpose == 1: # 90 CW in image -> 90 CCW around Z in camera
            R_z_ccw = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            T_ns = T_ns @ R_z_ccw
            
        # Invert to get World-to-Camera
        R_cw, t_cw = T_ns[:3, :3], T_ns[:3, 3]
        R_wc, t_wc = R_cw.T, -R_cw.T @ t_cw
        
        # Draw
        img = draw_grid(img, R_wc, t_wc, K, floor_y=floor_y, size=5.0)
        img = draw_axis(img, R_wc, t_wc, K, axis_length=0.5)
        
        cv2.putText(img, f"Frame: {idx} Time: {video_time:.2f}s", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imwrite(str(frame_path), img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect scan package integrity")
    parser.add_argument("input", help="Path to input .zip or directory")
    parser.add_argument("--out", default="debug_check", help="Output directory")
    parser.add_argument("--transpose", type=int, help="FFmpeg transpose: 1=90CW, 2=90CCW")
    
    args = parser.parse_args()
    input_path, output_dir = Path(args.input), Path(args.out)
    
    if input_path.suffix == '.zip':
        work_dir = Path(tempfile.mkdtemp())
        package_dir, _, _ = ingest(input_path, work_dir, validate=False)
    else:
        package_dir = input_path
        
    try:
        analyze_scan(package_dir, output_dir, transpose=args.transpose)
        print(f"\nResults in: {output_dir}")
    finally:
        if input_path.suffix == '.zip' and 'work_dir' in locals():
            shutil.rmtree(work_dir)
