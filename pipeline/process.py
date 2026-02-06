"""
Main Processing Pipeline Orchestrator

Coordinates the full pipeline from iOS scan package to web-ready tour package.
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass, field
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import typer

from .ingest import ingest, IngestError
from .extract_frames import extract_frames, ExtractionError
from .convert_poses import convert_from_manifest, ConversionError
from .train import train_and_export, TrainingConfig, TrainingError
from .compress import compress_ply_to_spz, CompressionError
from .collision import (
    generate_collision_from_usdz,
    generate_collision_from_floorplan,
    generate_floorplan_svg,
    CollisionError,
)
from .package import create_tour_package, create_zip_package, PackageError

console = Console()
app = typer.Typer(help="Reality Engine Processing Pipeline")


@dataclass
class PipelineConfig:
    """Configuration for the processing pipeline."""
    # Frame extraction
    target_fps: Optional[float] = None
    target_frame_count: int = 250  # Default: 250 frames for good coverage without OOM
    include_limited_tracking: bool = False  # Exclude "limited" tracking - they have bad poses

    # Training
    training_iterations: int = 30000
    skip_training: bool = False

    # Compression
    compression_quality: str = "high"

    # Collision
    simplify_collision: bool = True
    collision_target_faces: int = 10000

    # Output
    create_zip: bool = True


@dataclass
class PipelineStats:
    """Statistics collected during pipeline execution."""
    start_time: float = 0
    end_time: float = 0
    stages: Dict = field(default_factory=dict)

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def record_stage(self, name: str, duration: float, **kwargs):
        self.stages[name] = {"duration_seconds": duration, **kwargs}

    @property
    def total_duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> Dict:
        return {
            "total_duration_seconds": self.total_duration,
            "stages": self.stages,
        }


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    config: Optional[PipelineConfig] = None
) -> Path:
    """
    Run the complete processing pipeline.

    Args:
        input_path: Path to scan package (.zip or directory)
        output_dir: Output directory for all artifacts
        config: Pipeline configuration

    Returns:
        Path to final tour package
    """
    config = config or PipelineConfig()
    stats = PipelineStats()
    stats.start()

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "work"

    console.print(Panel.fit(
        "[bold blue]Reality Engine Processing Pipeline[/bold blue]\n"
        f"Input: {input_path}\n"
        f"Output: {output_dir}",
        border_style="blue"
    ))

    # Stage 1: Ingest
    console.print("\n[bold]Stage 1: Ingest[/bold]")
    stage_start = time.time()
    try:
        package_dir, manifest, package_info = ingest(input_path, work_dir)
        scan_id = manifest.scan_id
    except IngestError as e:
        console.print(f"[bold red]Ingestion failed:[/bold red] {e}")
        raise
    stats.record_stage("ingest", time.time() - stage_start,
                      scan_id=manifest.scan_id if manifest else None,
                      pose_count=len(manifest.poses) if manifest else 0)

    # Stage 2: Extract Frames
    console.print("\n[bold]Stage 2: Extract Frames[/bold]")
    stage_start = time.time()
    try:
        video_path = package_dir / manifest.assets.video_path
        frames_dir, matched_frames = extract_frames(
            video_path=video_path,
            poses=[p.model_dump() for p in manifest.poses],
            output_dir=work_dir / "extracted",
            target_fps=config.target_fps,
            include_limited_tracking=config.include_limited_tracking,
            target_frame_count=config.target_frame_count,
        )
    except ExtractionError as e:
        console.print(f"[bold red]Frame extraction failed:[/bold red] {e}")
        raise
    stats.record_stage("extract_frames", time.time() - stage_start,
                      frame_count=len(matched_frames))

    # Stage 3: Convert Poses
    console.print("\n[bold]Stage 3: Convert Poses[/bold]")
    stage_start = time.time()
    try:
        transforms_path = work_dir / "transforms.json"
        convert_from_manifest(
            manifest_path=package_dir / "scan_manifest.json",
            frames_dir=frames_dir,
            output_path=transforms_path,
            frames_manifest_path=work_dir / "extracted" / "frames_manifest.json"
        )
    except ConversionError as e:
        console.print(f"[bold red]Pose conversion failed:[/bold red] {e}")
        raise
    stats.record_stage("convert_poses", time.time() - stage_start)

    # Stage 4: Train Gaussian Splat
    if not config.skip_training:
        console.print("\n[bold]Stage 4: Train Gaussian Splat[/bold]")
        stage_start = time.time()
        try:
            # Prepare training data
            train_data_dir = work_dir / "train_data"
            train_data_dir.mkdir(exist_ok=True)

            # Copy transforms and link frames
            import shutil
            shutil.copy(transforms_path, train_data_dir / "transforms.json")
            frames_link = train_data_dir / "frames"
            if frames_link.exists() or frames_link.is_symlink():
                frames_link.unlink() if frames_link.is_symlink() else shutil.rmtree(frames_link)
            frames_link.symlink_to(frames_dir.resolve())

            training_config = TrainingConfig(
                max_iterations=config.training_iterations
            )
            ply_path = train_and_export(
                data_dir=train_data_dir,
                output_dir=work_dir / "training",
                config=training_config,
                experiment_name=scan_id
            )
        except TrainingError as e:
            console.print(f"[bold red]Training failed:[/bold red] {e}")
            raise
        stats.record_stage("train", time.time() - stage_start,
                          iterations=config.training_iterations)
    else:
        console.print("\n[bold]Stage 4: Training (SKIPPED)[/bold]")
        # Look for existing Gaussian splat PLY file (not LiDAR mesh)
        ply_path = None
        for candidate in work_dir.rglob("*.ply"):
            # Check if it's a Gaussian splat PLY (has f_dc_0 property)
            try:
                with open(candidate, 'rb') as f:
                    header = f.read(2000).decode('utf-8', errors='ignore')
                    if 'f_dc_0' in header or 'scale_0' in header:
                        ply_path = candidate
                        console.print(f"[yellow]Using existing Gaussian splat PLY: {ply_path}[/yellow]")
                        break
            except Exception:
                continue

        if ply_path is None:
            console.print("[yellow]No Gaussian splat PLY found, skipping compression[/yellow]")

    # Stage 5: Prepare Splat File
    # Note: SPZ compression requires Niantic's spz tool. For now, use PLY directly.
    # Most viewers (SuperSplat, etc.) support PLY format natively.
    console.print("\n[bold]Stage 5: Prepare Splat File[/bold]")
    stage_start = time.time()
    if config.skip_training and ply_path is None:
        # Create dummy PLY for testing
        console.print("[yellow]Creating dummy PLY for local testing[/yellow]")
        splat_path = work_dir / "scene.ply"
        splat_path.write_text('ply\nformat ascii 1.0\nelement vertex 0\nend_header\n')
        compression_stats = {'compression_ratio': 1, 'note': 'dummy for testing'}
    else:
        # Use PLY directly (no compression)
        splat_path = ply_path
        ply_size = ply_path.stat().st_size / (1024 * 1024)
        console.print(f"[green]Using PLY directly: {ply_path.name} ({ply_size:.1f} MB)[/green]")
        compression_stats = {'compression_ratio': 1, 'format': 'ply', 'size_mb': ply_size}
    stats.record_stage("prepare_splat", time.time() - stage_start,
                      compression_ratio=compression_stats.get('compression_ratio', 1))

    # Stage 6: Generate Collision Mesh
    console.print("\n[bold]Stage 6: Generate Collision Mesh[/bold]")
    stage_start = time.time()
    collision_path = None
    floorplan_path = None

    try:
        # Try USDZ first
        if manifest.assets.room_geometry:
            usdz_path = package_dir / manifest.assets.room_geometry
            if usdz_path.exists():
                collision_path, collision_stats = generate_collision_from_usdz(
                    usdz_path,
                    work_dir,
                    simplify=config.simplify_collision,
                    target_faces=config.collision_target_faces
                )

        # Try PLY mesh (LiDAR scan)
        if collision_path is None:
            mesh_path = None
            if hasattr(manifest.assets, 'mesh_path') and manifest.assets.mesh_path:
                mesh_path = package_dir / manifest.assets.mesh_path
            else:
                # Look for mesh.ply in package
                candidate = package_dir / "mesh.ply"
                if candidate.exists():
                    mesh_path = candidate

            if mesh_path and mesh_path.exists():
                from .collision import generate_collision_from_ply
                collision_path, collision_stats = generate_collision_from_ply(
                    mesh_path,
                    work_dir,
                    simplify=config.simplify_collision,
                    target_faces=config.collision_target_faces
                )

        # Use floorplan data as last resort
        if collision_path is None and manifest.floorplan and manifest.floorplan.walls:
            collision_path, collision_stats = generate_collision_from_floorplan(
                manifest.floorplan.model_dump(),
                work_dir
            )

        # Generate floorplan SVG
        if manifest.floorplan:
            floorplan_path = generate_floorplan_svg(
                manifest.floorplan.model_dump(),
                work_dir / "floorplan.svg"
            )

    except CollisionError as e:
        console.print(f"[yellow]Collision generation warning:[/yellow] {e}")

    stats.record_stage("collision", time.time() - stage_start)

    # Stage 7: Create Package
    console.print("\n[bold]Stage 7: Create Tour Package[/bold]")
    stage_start = time.time()
    try:
        package_path = create_tour_package(
            scan_id=scan_id,
            scan_manifest=json.loads(manifest.model_dump_json()),
            splat_path=splat_path,
            collision_path=collision_path,
            floorplan_path=floorplan_path,
            frames_dir=frames_dir,
            output_dir=output_dir,
            processing_stats=stats.to_dict()
        )

        if config.create_zip:
            create_zip_package(package_path)

    except PackageError as e:
        console.print(f"[bold red]Packaging failed:[/bold red] {e}")
        raise

    stats.record_stage("package", time.time() - stage_start)
    stats.stop()

    # Final summary
    console.print(Panel.fit(
        f"[bold green]Pipeline Complete![/bold green]\n\n"
        f"Scan ID: {scan_id}\n"
        f"Total time: {stats.total_duration:.1f}s\n"
        f"Output: {package_path}",
        border_style="green"
    ))

    return package_path


@app.command()
def main(
    input_path: Path = typer.Argument(..., help="Path to scan package (.zip or directory)"),
    output_dir: Path = typer.Option(Path("./output"), help="Output directory"),
    iterations: int = typer.Option(30000, help="Training iterations"),
    skip_training: bool = typer.Option(False, "--skip-training", help="Skip training stage"),
    target_fps: Optional[float] = typer.Option(None, help="Target FPS for frame extraction"),
    target_frames: int = typer.Option(250, help="Target number of frames (default: 250, use 0 for no limit)"),
    no_zip: bool = typer.Option(False, "--no-zip", help="Don't create ZIP archive"),
    compression: str = typer.Option("high", help="Compression quality: low, medium, high"),
):
    """
    Run the complete Reality Engine processing pipeline.

    Takes an iOS scan package and produces a web-ready tour package.

    Frame extraction defaults to 250 frames using movement-based selection.
    For larger spaces, increase --target-frames (e.g., 400-500).
    """
    config = PipelineConfig(
        training_iterations=iterations,
        skip_training=skip_training,
        target_fps=target_fps,
        target_frame_count=target_frames if target_frames > 0 else None,
        create_zip=not no_zip,
        compression_quality=compression,
    )

    try:
        run_pipeline(input_path, output_dir, config)
    except Exception as e:
        console.print(f"[bold red]Pipeline failed:[/bold red] {e}")
        raise typer.Exit(1)


@app.command("stages")
def list_stages():
    """List all pipeline stages."""
    stages = [
        ("1. Ingest", "Unzip and validate scan package"),
        ("2. Extract Frames", "Extract video frames and match to poses"),
        ("3. Convert Poses", "Convert ARKit poses to Nerfstudio format"),
        ("4. Train", "Train Gaussian Splat model"),
        ("5. Compress", "Compress PLY to SPZ format"),
        ("6. Collision", "Generate collision mesh and floorplan"),
        ("7. Package", "Bundle final tour package"),
    ]

    console.print("[bold]Pipeline Stages:[/bold]\n")
    for name, desc in stages:
        console.print(f"  [blue]{name}[/blue]: {desc}")


if __name__ == "__main__":
    app()
