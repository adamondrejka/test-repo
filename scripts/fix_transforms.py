#!/usr/bin/env python3
"""Fix transforms.json to only include existing frames."""

import json
from pathlib import Path
import sys

def fix_transforms(data_dir: str = "."):
    data_dir = Path(data_dir)
    transforms_path = data_dir / "transforms.json"
    frames_dir = data_dir / "frames"

    # Load current transforms
    with open(transforms_path) as f:
        data = json.load(f)

    # Get existing frames
    existing_frames = set(f.name for f in frames_dir.glob('*.jpg'))
    existing_frames.update(f.name for f in frames_dir.glob('*.png'))
    print(f"Existing frames: {len(existing_frames)}")

    # Filter to only existing frames
    new_frames = []
    for frame in data['frames']:
        fname = Path(frame['file_path']).name
        if fname in existing_frames:
            new_frames.append(frame)

    print(f"Matched frames: {len(new_frames)}")

    if len(new_frames) == 0:
        print("ERROR: No frames matched! Check file paths.")
        sys.exit(1)

    data['frames'] = new_frames

    # Backup original
    backup_path = data_dir / "transforms.json.bak"
    if not backup_path.exists():
        import shutil
        shutil.copy(transforms_path, backup_path)
        print(f"Backed up to {backup_path}")

    # Save
    with open(transforms_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Updated transforms.json with {len(new_frames)} frames")

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    fix_transforms(data_dir)
