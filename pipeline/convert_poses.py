"""
Pose Conversion Pipeline Stage

Converts ARKit camera poses to Nerfstudio transforms.json format.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from rich.console import Console
from PIL import Image

from utils.matrix import (
    row_major_to_matrix,
    arkit_to_nerfstudio,
    compute_camera_intrinsics_dict,
    validate_transform,
)

console = Console()


def get_actual_image_dimensions(frames_dir: Path) -> Optional[Tuple[int, int]]:
    """
    Get actual image dimensions from first frame.

    Returns:
        Tuple of (width, height) or None if no frames found
    """
    frames = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if frames:
        with Image.open(frames[0]) as img:
            return img.size  # (width, height)
    return None


class ConversionError(Exception):
    """Error during pose conversion."""
    pass


def create_nerfstudio_frame(
    image_path: str,
    transform_matrix: List[float],
    timestamp: Optional[float] = None
) -> Dict:
    """
    Create a single frame entry for Nerfstudio transforms.json.

    Args:
        image_path: Relative path to image file
        transform_matrix: 4x4 transform matrix (row-major, 16 elements)
        timestamp: Optional timestamp for the frame

    Returns:
        Dictionary in Nerfstudio frame format
    """
    # Convert ARKit matrix to numpy
    matrix = row_major_to_matrix(transform_matrix)

    # Validate
    if not validate_transform(matrix):
        raise ConversionError(f"Invalid transform matrix for {image_path}")

    # Convert to Nerfstudio convention
    ns_matrix = arkit_to_nerfstudio(matrix)

    frame = {
        "file_path": image_path,
        "transform_matrix": ns_matrix.tolist(),
    }

    if timestamp is not None:
        frame["timestamp"] = timestamp

    return frame


def create_transforms_json(
    frames: List[Dict],
    calibration: Dict,
    output_path: Path,
    frames_dir: Optional[Path] = None,
    aabb_scale: int = 16,
    camera_model: str = "OPENCV"
) -> Dict:
    """
    Create Nerfstudio-compatible transforms.json.

    Args:
        frames: List of frame dictionaries with image_path and transform_matrix
        calibration: Camera calibration dict with intrinsic_matrix, image_width, image_height
        output_path: Path to write transforms.json
        frames_dir: Optional directory containing frames (for dimension auto-detection)
        aabb_scale: Axis-aligned bounding box scale
        camera_model: Camera model (OPENCV, PINHOLE, etc.)

    Returns:
        The complete transforms dictionary
    """
    # Get camera intrinsics
    intrinsics = compute_camera_intrinsics_dict(
        calibration['intrinsic_matrix'],
        calibration['image_width'],
        calibration['image_height']
    )

    # Auto-detect actual image dimensions from frames if available
    # This fixes portrait/landscape orientation mismatches
    if frames_dir:
        actual_dims = get_actual_image_dimensions(frames_dir)
        if actual_dims:
            actual_w, actual_h = actual_dims
            meta_w, meta_h = intrinsics['w'], intrinsics['h']

            # Check if dimensions are swapped (portrait vs landscape)
            if (actual_w, actual_h) != (meta_w, meta_h):
                if (actual_w, actual_h) == (meta_h, meta_w):
                    # Dimensions are swapped - adjust intrinsics
                    console.print(f"[yellow]Detected dimension swap: metadata {meta_w}x{meta_h} vs actual {actual_w}x{actual_h}[/yellow]")
                    console.print(f"[yellow]Adjusting intrinsics for actual image orientation[/yellow]")
                    intrinsics['w'] = actual_w
                    intrinsics['h'] = actual_h
                    # Swap cx/cy to match new orientation
                    intrinsics['cx'], intrinsics['cy'] = intrinsics['cy'], intrinsics['cx']
                    intrinsics['fl_x'], intrinsics['fl_y'] = intrinsics['fl_y'], intrinsics['fl_x']
                else:
                    console.print(f"[yellow]Warning: Image dimensions mismatch - metadata {meta_w}x{meta_h} vs actual {actual_w}x{actual_h}[/yellow]")
                    # Use actual dimensions
                    intrinsics['w'] = actual_w
                    intrinsics['h'] = actual_h

    # Create transforms structure
    transforms = {
        "camera_model": camera_model,
        "fl_x": intrinsics['fl_x'],
        "fl_y": intrinsics['fl_y'],
        "cx": intrinsics['cx'],
        "cy": intrinsics['cy'],
        "w": intrinsics['w'],
        "h": intrinsics['h'],
        "aabb_scale": aabb_scale,
        "frames": []
    }

    # Convert each frame
    console.print(f"[blue]Converting {len(frames)} poses to Nerfstudio format...[/blue]")

    for frame_data in frames:
        try:
            frame = create_nerfstudio_frame(
                image_path=frame_data['image_path'],
                transform_matrix=frame_data['transform_matrix'],
                timestamp=frame_data.get('timestamp')
            )
            transforms['frames'].append(frame)
        except ConversionError as e:
            console.print(f"[yellow]Warning: {e}[/yellow]")
            continue

    # Write to file
    with open(output_path, 'w') as f:
        json.dump(transforms, f, indent=2)

    console.print(f"[green]Wrote {len(transforms['frames'])} frames to {output_path}[/green]")

    return transforms


def convert_from_manifest(
    manifest_path: Path,
    frames_dir: Path,
    output_path: Path,
    frames_manifest_path: Optional[Path] = None
) -> Dict:
    """
    Convert poses from iOS scan manifest to Nerfstudio format.

    Args:
        manifest_path: Path to scan_manifest.json
        frames_dir: Directory containing extracted frames
        output_path: Path to write transforms.json
        frames_manifest_path: Optional path to frames_manifest.json
            (if provided, uses frame-pose matching from extraction)

    Returns:
        The complete transforms dictionary
    """
    # Load scan manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    calibration = manifest['calibration']

    # Determine frame-pose mapping
    if frames_manifest_path and frames_manifest_path.exists():
        # Use pre-computed frame-pose matching
        with open(frames_manifest_path, 'r') as f:
            frames_data = json.load(f)

        frames = [
            {
                'image_path': f"./frames/{fd['image_path']}",
                'transform_matrix': fd['transform_matrix'],
                'timestamp': fd['pose_timestamp'],
            }
            for fd in frames_data
        ]
    else:
        # Assume 1:1 mapping between frames and poses
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        poses = manifest['poses']

        if len(frame_files) != len(poses):
            console.print(f"[yellow]Warning: {len(frame_files)} frames but "
                         f"{len(poses)} poses. Using minimum.[/yellow]")

        frames = []
        for i, (frame_file, pose) in enumerate(zip(frame_files, poses)):
            frames.append({
                'image_path': f"./frames/{frame_file.name}",
                'transform_matrix': pose['transform_matrix'],
                'timestamp': pose.get('timestamp'),
            })

    return create_transforms_json(frames, calibration, output_path, frames_dir=frames_dir)


def compute_scene_bounds(transforms: Dict) -> Dict:
    """
    Compute scene bounding box from transforms.

    Returns:
        Dictionary with center, extent, and scale
    """
    positions = []

    for frame in transforms['frames']:
        matrix = np.array(frame['transform_matrix'])
        position = matrix[:3, 3]
        positions.append(position)

    positions = np.array(positions)

    min_bound = positions.min(axis=0)
    max_bound = positions.max(axis=0)
    center = (min_bound + max_bound) / 2
    extent = max_bound - min_bound

    return {
        'center': center.tolist(),
        'min': min_bound.tolist(),
        'max': max_bound.tolist(),
        'extent': extent.tolist(),
        'diagonal': float(np.linalg.norm(extent)),
    }


def validate_transforms(transforms: Dict) -> List[str]:
    """
    Validate a transforms.json file.

    Returns:
        List of warning/error messages
    """
    issues = []

    # Check required fields
    required_fields = ['fl_x', 'fl_y', 'cx', 'cy', 'w', 'h', 'frames']
    for field in required_fields:
        if field not in transforms:
            issues.append(f"Missing required field: {field}")

    # Check frames
    frames = transforms.get('frames', [])
    if len(frames) < 10:
        issues.append(f"Too few frames: {len(frames)} (minimum 10 recommended)")

    # Check for valid transforms
    invalid_count = 0
    for i, frame in enumerate(frames):
        matrix = np.array(frame.get('transform_matrix', []))
        if matrix.shape != (4, 4):
            invalid_count += 1
            continue
        if not validate_transform(matrix):
            invalid_count += 1

    if invalid_count > 0:
        issues.append(f"{invalid_count} frames have invalid transforms")

    # Check image paths exist (if possible)
    # This would require the base path

    return issues


# CLI entry point
if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def main(
        work_dir: Path = typer.Argument(..., help="Work directory (from ingest/extract stages)"),
    ):
        """Convert ARKit poses to Nerfstudio transforms.json format.

        Expects work_dir to contain:
        - Package with scan_manifest.json
        - extracted/frames/ directory
        - extracted/frames_manifest.json
        """
        # Find manifest
        manifest_candidates = list(work_dir.rglob("scan_manifest.json"))
        if not manifest_candidates:
            console.print(f"[bold red]Error:[/bold red] No scan_manifest.json found in {work_dir}")
            raise typer.Exit(1)

        manifest_path = manifest_candidates[0]
        frames_dir = work_dir / "extracted" / "frames"
        frames_manifest = work_dir / "extracted" / "frames_manifest.json"
        output_path = work_dir / "transforms.json"

        if not frames_dir.exists():
            console.print(f"[bold red]Error:[/bold red] Frames directory not found: {frames_dir}")
            console.print("Run extract_frames first.")
            raise typer.Exit(1)

        try:
            transforms = convert_from_manifest(
                manifest_path,
                frames_dir,
                output_path,
                frames_manifest if frames_manifest.exists() else None
            )

            # Print scene bounds
            bounds = compute_scene_bounds(transforms)
            console.print(f"\n[bold]Scene Bounds:[/bold]")
            console.print(f"  Center: {bounds['center']}")
            console.print(f"  Extent: {bounds['extent']}")
            console.print(f"  Diagonal: {bounds['diagonal']:.2f}m")

            # Validate
            issues = validate_transforms(transforms)
            if issues:
                console.print(f"\n[yellow]Validation warnings:[/yellow]")
                for issue in issues:
                    console.print(f"  [yellow]• {issue}[/yellow]")

            console.print(f"\n[bold green]Conversion complete![/bold green]")

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1)

    @app.command("from-paths")
    def from_paths(
        manifest_path: Path = typer.Argument(..., help="Path to scan_manifest.json"),
        frames_dir: Path = typer.Argument(..., help="Directory with extracted frames"),
        output_path: Path = typer.Option(
            Path("transforms.json"),
            help="Output path for transforms.json"
        ),
        frames_manifest: Optional[Path] = typer.Option(
            None,
            help="Path to frames_manifest.json from extraction"
        ),
    ):
        """Convert ARKit poses from explicit paths."""
        try:
            transforms = convert_from_manifest(
                manifest_path,
                frames_dir,
                output_path,
                frames_manifest
            )

            # Print scene bounds
            bounds = compute_scene_bounds(transforms)
            console.print(f"\n[bold]Scene Bounds:[/bold]")
            console.print(f"  Center: {bounds['center']}")
            console.print(f"  Extent: {bounds['extent']}")
            console.print(f"  Diagonal: {bounds['diagonal']:.2f}m")

            # Validate
            issues = validate_transforms(transforms)
            if issues:
                console.print(f"\n[yellow]Validation warnings:[/yellow]")
                for issue in issues:
                    console.print(f"  [yellow]• {issue}[/yellow]")

            console.print(f"\n[bold green]Conversion complete![/bold green]")

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1)

    app()
