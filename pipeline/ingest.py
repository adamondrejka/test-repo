"""
Ingestion Pipeline Stage

Handles unzipping and validation of scan packages from iOS app.
"""

import json
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple
from rich.console import Console

from utils.validation import validate_scan_package, ScanManifest

console = Console()


class IngestError(Exception):
    """Error during ingestion process."""
    pass


def unzip_package(
    zip_path: Path,
    output_dir: Path,
    overwrite: bool = False
) -> Path:
    """
    Unzip a scan package to the specified directory.

    Args:
        zip_path: Path to the .zip file
        output_dir: Directory to extract to
        overwrite: If True, overwrite existing directory

    Returns:
        Path to the extracted package directory
    """
    if not zip_path.exists():
        raise IngestError(f"Zip file not found: {zip_path}")

    if not zipfile.is_zipfile(zip_path):
        raise IngestError(f"Not a valid zip file: {zip_path}")

    # Determine output directory name
    package_name = zip_path.stem
    package_dir = output_dir / package_name

    if package_dir.exists():
        if overwrite:
            console.print(f"[yellow]Removing existing directory: {package_dir}[/yellow]")
            shutil.rmtree(package_dir)
        else:
            raise IngestError(f"Output directory already exists: {package_dir}")

    # Extract
    console.print(f"[blue]Extracting {zip_path.name}...[/blue]")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)

    # Handle case where zip extracts to a subdirectory
    extracted_items = list(output_dir.iterdir())
    if len(extracted_items) == 1 and extracted_items[0].is_dir():
        # Contents extracted to single subdirectory
        actual_dir = extracted_items[0]
        if actual_dir != package_dir:
            actual_dir.rename(package_dir)

    if not package_dir.exists():
        # Try to find the manifest and use that directory
        for item in output_dir.rglob("scan_manifest.json"):
            package_dir = item.parent
            break

    console.print(f"[green]Extracted to: {package_dir}[/green]")
    return package_dir


def load_manifest(package_dir: Path) -> ScanManifest:
    """
    Load and parse the scan manifest from a package directory.

    Args:
        package_dir: Directory containing scan_manifest.json

    Returns:
        Parsed ScanManifest object
    """
    manifest_path = package_dir / "scan_manifest.json"

    if not manifest_path.exists():
        raise IngestError(f"Manifest not found: {manifest_path}")

    try:
        with open(manifest_path, 'r') as f:
            data = json.load(f)
        return ScanManifest(**data)
    except json.JSONDecodeError as e:
        raise IngestError(f"Invalid JSON in manifest: {e}")
    except Exception as e:
        raise IngestError(f"Failed to parse manifest: {e}")


def validate_package(package_dir: Path) -> Tuple[bool, dict]:
    """
    Validate a scan package directory.

    Args:
        package_dir: Directory containing extracted scan package

    Returns:
        Tuple of (is_valid, info_dict)
    """
    console.print("[blue]Validating scan package...[/blue]")

    is_valid, info, errors = validate_scan_package(package_dir)

    if errors:
        console.print("[red]Validation errors:[/red]")
        for error in errors:
            console.print(f"  [red]• {error}[/red]")

    if is_valid:
        console.print("[green]Package validation passed![/green]")

        # Print summary
        if 'poses' in info:
            poses = info['poses']
            console.print(f"  Poses: {poses['total_poses']}")
            console.print(f"  Duration: {poses['duration']:.1f}s")
            console.print(f"  Avg FPS: {poses['avg_fps']:.1f}")
            console.print(f"  Tracking: {poses['normal_tracking']} normal, "
                         f"{poses['limited_tracking']} limited")

        if 'video' in info:
            video = info['video']
            if 'width' in video:
                console.print(f"  Video: {video['width']}x{video['height']} @ {video.get('fps', 0):.1f}fps")

    return is_valid, info


def ingest(
    input_path: Path,
    work_dir: Path,
    validate: bool = True
) -> Tuple[Path, ScanManifest, dict]:
    """
    Full ingestion pipeline: unzip (if needed), validate, and load manifest.

    Args:
        input_path: Path to .zip file or extracted directory
        work_dir: Working directory for extraction
        validate: Whether to validate the package

    Returns:
        Tuple of (package_directory, manifest, package_info)
    """
    # Create work directory
    work_dir.mkdir(parents=True, exist_ok=True)

    # Determine if we need to unzip
    if input_path.suffix == '.zip':
        package_dir = unzip_package(input_path, work_dir, overwrite=True)
    elif input_path.is_dir():
        package_dir = input_path
    else:
        raise IngestError(f"Input must be a .zip file or directory: {input_path}")

    # Load manifest
    manifest = load_manifest(package_dir)
    console.print(f"[green]Loaded manifest for scan: {manifest.scan_id}[/green]")

    # Validate
    info = {}
    if validate:
        is_valid, info = validate_package(package_dir)
        if not is_valid:
            raise IngestError("Package validation failed")

    return package_dir, manifest, info


# CLI entry point
if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def main(
        input_path: Path = typer.Argument(..., help="Path to scan package (.zip or directory)"),
        work_dir: Path = typer.Option(Path("./work"), help="Working directory"),
        skip_validation: bool = typer.Option(False, "--no-validate", help="Skip validation")
    ):
        """Ingest and validate a scan package."""
        try:
            package_dir, manifest, info = ingest(
                input_path,
                work_dir,
                validate=not skip_validation
            )
            console.print(f"\n[bold green]Ingestion complete![/bold green]")
            console.print(f"Package: {package_dir}")
            console.print(f"Scan ID: {manifest.scan_id}")
            console.print(f"Poses: {len(manifest.poses)}")
        except IngestError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1)

    app()
