"""
Packaging Pipeline Stage

Bundles all outputs into a final tour package ready for web viewing.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from rich.console import Console

console = Console()


class PackageError(Exception):
    """Error during package creation."""
    pass


def generate_thumbnail(
    frames_dir: Path,
    output_path: Path,
    size: tuple = (400, 300)
) -> Path:
    """
    Generate a thumbnail from the sharpest, best-lit frame.

    Scores up to 20 evenly-spaced candidate frames by sharpness and
    brightness balance, picking the best one instead of the first frame.

    Args:
        frames_dir: Directory containing frames
        output_path: Output path for thumbnail
        size: Thumbnail size (width, height)

    Returns:
        Path to generated thumbnail
    """
    frames = sorted(frames_dir.glob("frame_*.jpg")) + sorted(frames_dir.glob("frame_*.png"))

    if not frames:
        raise PackageError("No frames found for thumbnail")

    # Score up to 20 evenly-spaced frames
    candidates = frames[::max(1, len(frames) // 20)]
    best_score = -1
    best_frame = candidates[0]

    try:
        import cv2
        import numpy as np

        for frame_path in candidates:
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Brightness (penalize too dark or too bright)
            brightness = gray.mean()
            brightness_score = 1.0 - abs(brightness - 127) / 127
            score = sharpness * brightness_score
            if score > best_score:
                best_score = score
                best_frame = frame_path

        console.print(f"[green]Selected thumbnail: {best_frame.name} "
                      f"(score={best_score:.0f})[/green]")
    except ImportError:
        console.print("[yellow]cv2 not available, using first frame for thumbnail[/yellow]")

    try:
        with Image.open(best_frame) as img:
            img.thumbnail(size, Image.Resampling.LANCZOS)
            output_path = output_path.with_suffix('.jpg')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, 'JPEG', quality=85)

        console.print(f"[green]Generated thumbnail: {output_path}[/green]")
        return output_path
    except Exception as e:
        raise PackageError(f"Failed to generate thumbnail: {e}")


def create_metadata(
    scan_id: str,
    scan_manifest: Dict,
    processing_stats: Dict,
    output_path: Path,
    splat_filename: str = "scene.ply",
    lod_levels: Optional[List[Dict]] = None,
) -> Path:
    """
    Create metadata.json for the tour package.

    Args:
        scan_id: Unique scan identifier
        scan_manifest: Original scan manifest data
        processing_stats: Statistics from processing stages
        output_path: Output path for metadata.json

    Returns:
        Path to metadata file
    """
    metadata = {
        "version": "1.0",
        "scan_id": scan_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "original_scan": {
            "timestamp": scan_manifest.get("timestamp"),
            "scan_type": scan_manifest.get("scan_type", "INDOOR_HYBRID"),
            "calibration": {
                "width": scan_manifest.get("calibration", {}).get("image_width"),
                "height": scan_manifest.get("calibration", {}).get("image_height"),
            },
            "frame_count": len(scan_manifest.get("poses", [])),
        },
        "assets": {
            "splat": {
                "file": splat_filename,
                "lod_levels": lod_levels or [],
            },
            "collision": "collision.glb",
            "floorplan": "floorplan.svg",
            "thumbnail": "thumbnail.jpg",
        },
        "processing": processing_stats,
        "viewer": {
            "start_position": None,  # Could be calculated from poses
            "camera_height": 1.6,  # Standard eye height
            "movement_speed": 2.0,
            "collision_enabled": True,
        }
    }

    # Calculate start position from first pose
    poses = scan_manifest.get("poses", [])
    if poses:
        first_pose = poses[0].get("transform_matrix", [])
        if len(first_pose) == 16:
            metadata["viewer"]["start_position"] = [
                first_pose[3],   # X (column 3, row 0)
                first_pose[7],   # Y (column 3, row 1)
                first_pose[11],  # Z (column 3, row 2)
            ]

    # Calculate bounds from floorplan
    floorplan = scan_manifest.get("floorplan") or {}
    if floorplan.get("walls"):
        walls = floorplan["walls"]
        all_points = []
        for wall in walls:
            all_points.append(wall["start"])
            all_points.append(wall["end"])

        if all_points:
            xs = [p[0] for p in all_points]
            zs = [p[1] for p in all_points]
            metadata["bounds"] = {
                "min": [min(xs), 0, min(zs)],
                "max": [max(xs), 3.0, max(zs)],  # Assume 3m ceiling
            }

    output_path = output_path.with_name("metadata.json")
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    console.print(f"[green]Created metadata: {output_path}[/green]")
    return output_path


def create_tour_package(
    scan_id: str,
    scan_manifest: Dict,
    splat_path: Path,
    collision_path: Optional[Path],
    floorplan_path: Optional[Path],
    frames_dir: Path,
    output_dir: Path,
    processing_stats: Optional[Dict] = None,
    lod_levels: Optional[List[Dict]] = None,
    ply_path: Optional[Path] = None,
) -> Path:
    """
    Create a complete tour package.

    Package structure:
    tour_package/
    ├── metadata.json
    ├── scene.spz
    ├── collision.glb (optional)
    ├── floorplan.svg (optional)
    └── thumbnail.jpg

    Args:
        scan_id: Unique scan identifier
        scan_manifest: Original scan manifest
        splat_path: Path to compressed splat file (.spz)
        collision_path: Optional path to collision mesh (.glb)
        floorplan_path: Optional path to floorplan (.svg)
        frames_dir: Directory with extracted frames (for thumbnail)
        output_dir: Output directory for package
        processing_stats: Optional processing statistics

    Returns:
        Path to created package directory
    """
    package_dir = output_dir / f"tour_{scan_id}"
    package_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[blue]Creating tour package: {package_dir}[/blue]")

    # Copy splat file (supports both .ply and .spz)
    if not splat_path.exists():
        raise PackageError(f"Splat file not found: {splat_path}")

    splat_ext = splat_path.suffix.lower()
    splat_output_name = f"scene{splat_ext}"
    shutil.copy(splat_path, package_dir / splat_output_name)
    console.print(f"  Copied {splat_output_name}")

    # Always include PLY alongside SPZ for safety
    if ply_path and ply_path.exists() and ply_path != splat_path:
        ply_output_name = f"scene{ply_path.suffix.lower()}"
        if ply_output_name != splat_output_name:
            shutil.copy(ply_path, package_dir / ply_output_name)
            console.print(f"  Copied {ply_output_name}")

    # Copy LOD files if available (both PLY and SPZ)
    if lod_levels:
        lod_source_dir = splat_path.parent / "lod"
        for lod in lod_levels:
            # Always copy PLY
            ply_file = lod["file"]
            ply_src = lod_source_dir / ply_file
            if ply_src.exists():
                shutil.copy(ply_src, package_dir / ply_file)
                console.print(f"  Copied {ply_file}")
            # Also copy SPZ if available
            if "file_spz" in lod:
                spz_file = lod["file_spz"]
                spz_src = lod_source_dir / spz_file
                if spz_src.exists():
                    shutil.copy(spz_src, package_dir / spz_file)
                    console.print(f"  Copied {spz_file}")

    # Copy collision mesh if available
    if collision_path and collision_path.exists():
        shutil.copy(collision_path, package_dir / "collision.glb")
        console.print("  Copied collision.glb")

    # Copy floorplan if available
    if floorplan_path and floorplan_path.exists():
        shutil.copy(floorplan_path, package_dir / "floorplan.svg")
        console.print("  Copied floorplan.svg")

    # Generate thumbnail
    try:
        generate_thumbnail(frames_dir, package_dir / "thumbnail.jpg")
    except PackageError as e:
        console.print(f"[yellow]Warning: {e}[/yellow]")

    # Create metadata
    create_metadata(
        scan_id,
        scan_manifest,
        processing_stats or {},
        package_dir / "metadata.json",
        splat_filename=splat_output_name,
        lod_levels=lod_levels,
    )

    # Calculate total package size
    total_size = sum(f.stat().st_size for f in package_dir.glob("*"))
    console.print(f"\n[bold green]Package created: {package_dir}[/bold green]")
    console.print(f"  Total size: {total_size / (1024*1024):.1f} MB")

    # List contents
    console.print("  Contents:")
    for f in sorted(package_dir.glob("*")):
        size_kb = f.stat().st_size / 1024
        console.print(f"    {f.name}: {size_kb:.1f} KB")

    return package_dir


def create_zip_package(
    package_dir: Path,
    output_path: Optional[Path] = None
) -> Path:
    """
    Create a ZIP archive of the tour package.

    Args:
        package_dir: Directory containing the package
        output_path: Optional output path for ZIP (defaults to package_dir.zip)

    Returns:
        Path to created ZIP file
    """
    if output_path is None:
        output_path = package_dir.with_suffix('.zip')

    output_path = output_path.with_suffix('.zip')

    shutil.make_archive(
        str(output_path.with_suffix('')),
        'zip',
        package_dir.parent,
        package_dir.name
    )

    console.print(f"[green]Created ZIP: {output_path}[/green]")
    console.print(f"  Size: {output_path.stat().st_size / (1024*1024):.1f} MB")

    return output_path


def validate_package(package_dir: Path) -> List[str]:
    """
    Validate a tour package has all required files.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check for splat file (either .ply or .spz)
    has_splat = (package_dir / "scene.ply").exists() or (package_dir / "scene.spz").exists()
    if not has_splat:
        errors.append("Missing splat file (scene.ply or scene.spz)")

    if not (package_dir / "metadata.json").exists():
        errors.append("Missing required file: metadata.json")

    # Validate metadata
    metadata_path = package_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            if 'scan_id' not in metadata:
                errors.append("Metadata missing scan_id")
            if 'assets' not in metadata:
                errors.append("Metadata missing assets")

        except json.JSONDecodeError as e:
            errors.append(f"Invalid metadata JSON: {e}")

    return errors


# CLI entry point
if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def main(
        work_dir: Path = typer.Argument(..., help="Work directory with processed assets"),
        output_dir: Path = typer.Argument(..., help="Output directory for package"),
        create_zip: bool = typer.Option(True, help="Create ZIP archive"),
    ):
        """Create a tour package from work directory.

        Expects work_dir to contain:
        - Package with scan_manifest.json
        - extracted/frames/ directory
        - scene.spz (compressed splat)
        - collision.glb (optional)
        - floorplan.svg (optional)
        """
        # Find manifest
        manifest_candidates = list(work_dir.rglob("scan_manifest.json"))
        if not manifest_candidates:
            console.print(f"[bold red]Error:[/bold red] No scan_manifest.json found in {work_dir}")
            raise typer.Exit(1)

        manifest_path = manifest_candidates[0]
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        scan_id = manifest.get('scan_id', 'unknown')

        # Find assets
        spz_path = work_dir / "scene.spz"
        if not spz_path.exists():
            console.print(f"[bold red]Error:[/bold red] scene.spz not found in {work_dir}")
            raise typer.Exit(1)

        collision_path = work_dir / "collision.glb"
        if not collision_path.exists():
            collision_path = None

        floorplan_path = work_dir / "floorplan.svg"
        if not floorplan_path.exists():
            floorplan_path = None

        frames_dir = work_dir / "extracted" / "frames"
        if not frames_dir.exists():
            frames_dir = work_dir

        try:
            package_dir = create_tour_package(
                scan_id=scan_id,
                scan_manifest=manifest,
                splat_path=spz_path,
                collision_path=collision_path,
                floorplan_path=floorplan_path,
                frames_dir=frames_dir,
                output_dir=output_dir,
            )

            # Validate
            errors = validate_package(package_dir)
            if errors:
                console.print("[yellow]Validation warnings:[/yellow]")
                for error in errors:
                    console.print(f"  [yellow]• {error}[/yellow]")

            # Create ZIP
            if create_zip:
                create_zip_package(package_dir)

        except PackageError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1)

    @app.command("from-paths")
    def from_paths(
        scan_id: str = typer.Argument(..., help="Scan identifier"),
        manifest_path: Path = typer.Argument(..., help="Path to scan_manifest.json"),
        splat_path: Path = typer.Argument(..., help="Path to scene.spz"),
        output_dir: Path = typer.Option(Path("./output"), help="Output directory"),
        collision_path: Optional[Path] = typer.Option(None, help="Path to collision.glb"),
        floorplan_path: Optional[Path] = typer.Option(None, help="Path to floorplan.svg"),
        frames_dir: Optional[Path] = typer.Option(None, help="Frames directory for thumbnail"),
        create_zip: bool = typer.Option(True, help="Create ZIP archive"),
    ):
        """Create a tour package from explicit asset paths."""
        # Load manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        try:
            package_dir = create_tour_package(
                scan_id=scan_id,
                scan_manifest=manifest,
                splat_path=splat_path,
                collision_path=collision_path,
                floorplan_path=floorplan_path,
                frames_dir=frames_dir or Path("."),
                output_dir=output_dir,
            )

            # Validate
            errors = validate_package(package_dir)
            if errors:
                console.print("[yellow]Validation warnings:[/yellow]")
                for error in errors:
                    console.print(f"  [yellow]• {error}[/yellow]")

            # Create ZIP
            if create_zip:
                create_zip_package(package_dir)

        except PackageError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1)

    @app.command("validate")
    def validate(
        package_dir: Path = typer.Argument(..., help="Package directory to validate"),
    ):
        """Validate a tour package."""
        errors = validate_package(package_dir)

        if errors:
            console.print("[red]Validation failed:[/red]")
            for error in errors:
                console.print(f"  [red]• {error}[/red]")
            raise typer.Exit(1)
        else:
            console.print("[green]Package is valid![/green]")

    app()
