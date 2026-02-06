#!/usr/bin/env python3
"""Debug script to analyze frames_manifest.json"""

import json
import sys
from pathlib import Path

def analyze_manifest(manifest_path: str):
    with open(manifest_path) as f:
        data = json.load(f)

    print("=" * 60)
    print("FRAMES_MANIFEST.JSON ANALYSIS")
    print("=" * 60)
    print(f"Total frames in manifest: {len(data)}")
    print()

    # Check for [0,0,0] positions
    zero_positions = []
    valid_positions = []

    print("FIRST 10 FRAMES:")
    print("-" * 60)

    for i, frame in enumerate(data[:10]):
        tm = frame.get('transform_matrix', [])
        if len(tm) == 16:
            # Position from row-major 4x4 matrix
            pos = [tm[3], tm[7], tm[11]]
            tracking = frame.get('tracking_state', 'unknown')

            is_zero = abs(pos[0]) < 0.001 and abs(pos[1]) < 0.001 and abs(pos[2]) < 0.001

            print(f"Frame {i}: {frame.get('image_path')}")
            print(f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            print(f"  Tracking: {tracking}")
            print(f"  Pose timestamp: {frame.get('pose_timestamp', 'N/A')}")
            print(f"  Frame timestamp: {frame.get('timestamp', 'N/A')}")
            if is_zero:
                print(f"  ⚠️  ZERO POSITION!")
            print()

    # Count statistics
    for frame in data:
        tm = frame.get('transform_matrix', [])
        if len(tm) == 16:
            pos = [tm[3], tm[7], tm[11]]
            is_zero = abs(pos[0]) < 0.001 and abs(pos[1]) < 0.001 and abs(pos[2]) < 0.001
            if is_zero:
                zero_positions.append(frame)
            else:
                valid_positions.append(frame)

    print("=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"Valid positions: {len(valid_positions)}")
    print(f"Zero positions:  {len(zero_positions)} ⚠️" if zero_positions else f"Zero positions:  0 ✓")
    print()

    # Check tracking states
    tracking_counts = {}
    for frame in data:
        state = frame.get('tracking_state', 'unknown')
        tracking_counts[state] = tracking_counts.get(state, 0) + 1

    print("TRACKING STATES:")
    for state, count in sorted(tracking_counts.items()):
        print(f"  {state}: {count}")
    print()

    # Check timestamp ranges
    timestamps = [f.get('timestamp', 0) for f in data if f.get('timestamp')]
    pose_timestamps = [f.get('pose_timestamp', 0) for f in data if f.get('pose_timestamp')]

    if timestamps:
        print("TIMESTAMP ANALYSIS:")
        print(f"  Frame timestamps: {min(timestamps):.3f} - {max(timestamps):.3f}")
        if pose_timestamps:
            print(f"  Pose timestamps:  {min(pose_timestamps):.3f} - {max(pose_timestamps):.3f}")
        print()

    # Show some valid and some zero position frames
    if zero_positions:
        print("=" * 60)
        print("SAMPLE ZERO-POSITION FRAMES")
        print("=" * 60)
        for frame in zero_positions[:3]:
            print(f"  {frame.get('image_path')} - tracking: {frame.get('tracking_state')}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/output/work/extracted/frames_manifest.json"
    analyze_manifest(path)
