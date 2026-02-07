import json
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def plot_trajectory(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frames = data['frames']
    timestamps = []
    positions = []
    
    print(f"Loading {len(frames)} frames...")

    for f in frames:
        # Extract timestamp (if available, else frame index)
        timestamps.append(f.get('timestamp', 0))
        
        # Extract position
        mat = np.array(f['transform_matrix'])
        positions.append(mat[:3, 3])
        
    positions = np.array(positions)
    
    # Calculate velocity to detect "stuck" frames
    velocities = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    stuck_frames = np.sum(velocities < 0.001) # Less than 1mm movement
    
    print(f"Analysis:")
    print(f"Total Frames: {len(frames)}")
    print(f"Stuck Frames (Zero Movement): {stuck_frames}")
    if stuck_frames > 5:
        print(f"⚠️  WARNING: High number of static frames! Sync issue likely.")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Top-down view (X-Z)
    ax1.plot(positions[:, 0], positions[:, 2], '-b', alpha=0.5)
    ax1.scatter(positions[0, 0], positions[0, 2], c='g', s=100, label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 2], c='r', s=100, label='End')
    ax1.set_title("Top-Down Path (X-Z)")
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Z (meters)")
    ax1.legend()
    ax1.grid(True)
    
    # Height over Time (Frame - Y)
    ax2.plot(range(len(positions)), positions[:, 1], '-r')
    ax2.set_title("Height Profile (Y)")
    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("Height (meters)")
    ax2.grid(True)
    
    output_img = Path(json_path).parent / "trajectory_debug.png"
    plt.savefig(output_img)
    print(f"\nGraph saved to: {output_img}")
    print("Check this image to see if the path is smooth or knotty.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_trajectory.py <path_to_transforms.json>")
        sys.exit(1)
    
    plot_trajectory(sys.argv[1])
