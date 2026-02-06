#!/usr/bin/env python3
"""Debug script to analyze camera poses in transforms.json"""

import json
import numpy as np
from pathlib import Path
import sys

def analyze_transforms(transforms_path: str):
    with open(transforms_path) as f:
        data = json.load(f)

    print("=" * 60)
    print("TRANSFORMS.JSON ANALYSIS")
    print("=" * 60)
    print()
    print(f"Frames: {len(data['frames'])}")
    print(f"Image size: {data.get('w', '?')} x {data.get('h', '?')}")
    print(f"Focal length: fx={data.get('fl_x', 0):.1f}, fy={data.get('fl_y', 0):.1f}")
    print(f"Principal point: cx={data.get('cx', 0):.1f}, cy={data.get('cy', 0):.1f}")
    print(f"Camera model: {data.get('camera_model', '?')}")
    print(f"AABB scale: {data.get('aabb_scale', '?')}")
    print()

    # Analyze camera positions and orientations
    positions = []
    look_dirs = []
    up_dirs = []

    print("=" * 60)
    print("FIRST 5 CAMERA POSES")
    print("=" * 60)

    for i, frame in enumerate(data['frames'][:5]):
        m = np.array(frame['transform_matrix'])
        pos = m[:3, 3]  # Translation = camera position

        # Camera axes in world space
        right = m[:3, 0]  # X axis
        up = m[:3, 1]     # Y axis
        look = m[:3, 2]   # Z axis (camera looks along -Z in OpenGL)

        positions.append(pos)
        look_dirs.append(look)
        up_dirs.append(up)

        print(f"\nFrame {i}: {frame['file_path']}")
        print(f"  Position:  [{pos[0]:8.3f}, {pos[1]:8.3f}, {pos[2]:8.3f}]")
        print(f"  Right (X): [{right[0]:8.3f}, {right[1]:8.3f}, {right[2]:8.3f}]")
        print(f"  Up (Y):    [{up[0]:8.3f}, {up[1]:8.3f}, {up[2]:8.3f}]")
        print(f"  Z axis:    [{look[0]:8.3f}, {look[1]:8.3f}, {look[2]:8.3f}]")

        # Check if rotation is valid
        det = np.linalg.det(m[:3, :3])
        print(f"  Rotation det: {det:.4f} (should be 1.0)")

    # Analyze all poses
    all_positions = []
    for frame in data['frames']:
        m = np.array(frame['transform_matrix'])
        all_positions.append(m[:3, 3])

    all_positions = np.array(all_positions)

    print()
    print("=" * 60)
    print("ALL CAMERAS STATISTICS")
    print("=" * 60)
    print(f"Min position:  [{all_positions.min(axis=0)[0]:.3f}, {all_positions.min(axis=0)[1]:.3f}, {all_positions.min(axis=0)[2]:.3f}]")
    print(f"Max position:  [{all_positions.max(axis=0)[0]:.3f}, {all_positions.max(axis=0)[1]:.3f}, {all_positions.max(axis=0)[2]:.3f}]")
    print(f"Spread:        [{(all_positions.max(axis=0) - all_positions.min(axis=0))[0]:.3f}, {(all_positions.max(axis=0) - all_positions.min(axis=0))[1]:.3f}, {(all_positions.max(axis=0) - all_positions.min(axis=0))[2]:.3f}]")
    print(f"Center:        [{all_positions.mean(axis=0)[0]:.3f}, {all_positions.mean(axis=0)[1]:.3f}, {all_positions.mean(axis=0)[2]:.3f}]")

    # Check for potential issues
    print()
    print("=" * 60)
    print("POTENTIAL ISSUES")
    print("=" * 60)

    spread = all_positions.max(axis=0) - all_positions.min(axis=0)
    if spread.max() < 0.1:
        print("⚠️  Camera spread very small (<0.1m) - cameras might be at same position")
    elif spread.max() > 100:
        print("⚠️  Camera spread very large (>100m) - scale might be wrong")
    else:
        print(f"✓ Camera spread looks reasonable: {spread.max():.2f}m")

    # Check Y values (should be roughly constant for indoor scan at eye level)
    y_spread = all_positions[:, 1].max() - all_positions[:, 1].min()
    y_mean = all_positions[:, 1].mean()
    print(f"✓ Y (height) spread: {y_spread:.2f}m, mean: {y_mean:.2f}m")

    # Check for NaN or Inf
    if np.any(np.isnan(all_positions)) or np.any(np.isinf(all_positions)):
        print("❌ Found NaN or Inf values in positions!")
    else:
        print("✓ No NaN or Inf values")

    # Sample raw matrix for inspection
    print()
    print("=" * 60)
    print("RAW MATRIX (first frame)")
    print("=" * 60)
    m = np.array(data['frames'][0]['transform_matrix'])
    print(m)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/output/work/transforms.json"
    analyze_transforms(path)
