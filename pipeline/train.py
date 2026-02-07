"""
Gaussian Splat Training Pipeline Stage

Trains a Gaussian Splat model using Nerfstudio/gsplat.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@dataclass
class TrainingConfig:
    """Configuration for Gaussian Splat training."""
    max_iterations: int = 30000
    warmup_iterations: int = 500
    densify_until: int = 15000
    densify_interval: int = 100
    densify_grad_thresh: float = 0.0002
    cull_alpha_thresh: float = 0.005
    split_size_thresh: float = 0.01
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2

    def to_nerfstudio_args(self) -> List[str]:
        """Convert to nerfstudio CLI arguments."""
        # Note: nerfstudio uses top-level --max-num-iterations, not --pipeline.model.*
        return [
            '--max-num-iterations', str(self.max_iterations),
        ]


class TrainingError(Exception):
    """Error during training."""
    pass


def check_nerfstudio_installed() -> bool:
    """Check if nerfstudio is available."""
    try:
        result = subprocess.run(
            ['ns-train', '--help'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_gpu_available() -> Dict:
    """Check CUDA GPU availability."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpus = []
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 3:
                    gpus.append({
                        'name': parts[0],
                        'memory_total': parts[1],
                        'memory_free': parts[2],
                    })
            return {'available': True, 'gpus': gpus}
    except FileNotFoundError:
        pass

    return {'available': False, 'gpus': []}


def prepare_training_data(
    transforms_path: Path,
    frames_dir: Path,
    output_dir: Path
) -> Path:
    """
    Prepare data directory structure for Nerfstudio training.

    Expected structure:
    output_dir/
    ├── transforms.json
    └── frames/
        ├── frame_000001.jpg
        └── ...
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy transforms.json
    shutil.copy(transforms_path, output_dir / "transforms.json")

    # Link or copy frames directory
    target_frames = output_dir / "frames"
    if target_frames.exists():
        shutil.rmtree(target_frames)

    # Use symlink if possible, otherwise copy
    try:
        target_frames.symlink_to(frames_dir.resolve())
    except OSError:
        shutil.copytree(frames_dir, target_frames)

    console.print(f"[green]Prepared training data in {output_dir}[/green]")
    return output_dir


def train_gaussian_splat(
    data_dir: Path,
    output_dir: Path,
    config: Optional[TrainingConfig] = None,
    experiment_name: str = "reality_scan",
    use_viewer: bool = False,
    verbose: bool = True
) -> Path:
    """
    Train a Gaussian Splat model using Nerfstudio.

    Args:
        data_dir: Directory containing transforms.json and frames/
        output_dir: Directory for training outputs
        config: Training configuration
        experiment_name: Name for the experiment
        use_viewer: Whether to launch the viewer during training
        verbose: Show training output

    Returns:
        Path to the trained model directory
    """
    if not check_nerfstudio_installed():
        raise TrainingError("Nerfstudio is not installed. Install with: pip install nerfstudio")

    gpu_info = check_gpu_available()
    if not gpu_info['available']:
        console.print("[yellow]Warning: No GPU detected. Training will be slow.[/yellow]")
    else:
        for gpu in gpu_info['gpus']:
            console.print(f"[blue]GPU: {gpu['name']} ({gpu['memory_free']} free)[/blue]")

    config = config or TrainingConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log Nerfstudio version for debugging
    try:
        ver = subprocess.run(['pip', 'show', 'nerfstudio'], capture_output=True, text=True)
        for line in ver.stdout.splitlines():
            if line.startswith('Version:'):
                console.print(f"[dim]Nerfstudio {line}[/dim]")
                break
    except Exception:
        pass

    # Log first frame convention for debugging
    transforms_file = data_dir / "transforms.json"
    if transforms_file.exists():
        import json as _json
        with open(transforms_file) as _f:
            _tf = _json.load(_f)
        if _tf.get('frames'):
            _m = _tf['frames'][0]['transform_matrix']
            _y_col = [_m[0][1], _m[1][1], _m[2][1]]
            console.print(f"[dim]Camera Y column (frame 0): {_y_col}[/dim]")
            console.print(f"[dim]  Y>0 = OpenGL/ARKit, Y<0 = OpenCV[/dim]")
        console.print(f"[dim]Frames: {len(_tf.get('frames', []))}, "
                      f"aabb_scale: {_tf.get('aabb_scale')}[/dim]")

    # Build command — training args BEFORE 'nerfstudio-data', dataparser args AFTER
    cmd = [
        'ns-train', 'splatfacto',
        '--output-dir', str(output_dir),
        '--experiment-name', experiment_name,
        '--timestamp', 'latest',
        '--pipeline.model.camera-optimizer.mode', 'SO3xR3',
    ]

    # Add config arguments (must be before nerfstudio-data subcommand)
    cmd.extend(config.to_nerfstudio_args())

    # Viewer settings - use 'tensorboard' for logging without external service
    if not use_viewer:
        cmd.extend(['--viewer.quit-on-train-completion', 'True'])
        cmd.extend(['--vis', 'tensorboard'])

    # Dataparser subcommand and its args go LAST
    cmd.extend(['nerfstudio-data', '--data', str(data_dir)])

    console.print(f"[blue]Starting Gaussian Splat training...[/blue]")
    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")

    # Run training
    try:
        if verbose:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in process.stdout:
                # Parse progress from output
                if 'Iteration' in line or 'iter' in line.lower():
                    console.print(f"[dim]{line.strip()}[/dim]")
                elif 'error' in line.lower():
                    console.print(f"[red]{line.strip()}[/red]")
                elif 'warning' in line.lower():
                    console.print(f"[yellow]{line.strip()}[/yellow]")

            process.wait()
            if process.returncode != 0:
                raise TrainingError(f"Training failed with return code {process.returncode}")
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise TrainingError(f"Training failed: {result.stderr}")

    except subprocess.CalledProcessError as e:
        raise TrainingError(f"Training failed: {e}")

    # Find output model
    model_dir = output_dir / experiment_name / "splatfacto" / "latest"
    if not model_dir.exists():
        # Try to find it
        for path in output_dir.rglob("config.yml"):
            model_dir = path.parent
            break

    if not model_dir.exists():
        raise TrainingError(f"Could not find trained model in {output_dir}")

    console.print(f"[green]Training complete! Model saved to {model_dir}[/green]")
    return model_dir


def export_splat(
    model_dir: Path,
    output_path: Path,
    export_format: str = "ply"
) -> Path:
    """
    Export trained model to PLY or other format.

    Args:
        model_dir: Directory containing trained model
        output_path: Path for exported file
        export_format: Export format (ply, ckpt)

    Returns:
        Path to exported file
    """
    output_path = output_path.with_suffix(f".{export_format}")

    cmd = [
        'ns-export', 'gaussian-splat',
        '--load-config', str(model_dir / "config.yml"),
        '--output-dir', str(output_path.parent),
    ]

    console.print(f"[blue]Exporting model to {output_path}...[/blue]")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise TrainingError(f"Export failed: {e.stderr}")

    # Find the exported file
    exported_files = list(output_path.parent.glob("*.ply"))
    if exported_files:
        # Rename to desired output path
        shutil.move(exported_files[0], output_path)
        console.print(f"[green]Exported to {output_path}[/green]")
        return output_path

    raise TrainingError("Export completed but could not find output file")


def train_and_export(
    data_dir: Path,
    output_dir: Path,
    config: Optional[TrainingConfig] = None,
    experiment_name: str = "reality_scan"
) -> Path:
    """
    Complete training pipeline: train and export to PLY.

    Args:
        data_dir: Directory containing transforms.json and frames/
        output_dir: Directory for all outputs
        config: Training configuration
        experiment_name: Name for the experiment

    Returns:
        Path to exported PLY file
    """
    # Train
    model_dir = train_gaussian_splat(
        data_dir,
        output_dir / "training",
        config=config,
        experiment_name=experiment_name
    )

    # Export
    ply_path = export_splat(
        model_dir,
        output_dir / "splat.ply"
    )

    return ply_path


# CLI entry point
if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def main(
        data_dir: Path = typer.Argument(..., help="Directory with transforms.json and frames/"),
        output_dir: Path = typer.Option(Path("./output"), help="Output directory"),
        iterations: int = typer.Option(30000, help="Training iterations"),
        experiment_name: str = typer.Option("reality_scan", help="Experiment name"),
        export_only: bool = typer.Option(False, help="Only export existing model"),
        model_dir: Optional[Path] = typer.Option(None, help="Model dir for export-only"),
    ):
        """Train a Gaussian Splat model."""
        try:
            if export_only:
                if not model_dir:
                    console.print("[red]--model-dir required for export-only[/red]")
                    raise typer.Exit(1)
                export_splat(model_dir, output_dir / "splat.ply")
            else:
                config = TrainingConfig(max_iterations=iterations)
                ply_path = train_and_export(
                    data_dir,
                    output_dir,
                    config=config,
                    experiment_name=experiment_name
                )
                console.print(f"\n[bold green]Complete! Output: {ply_path}[/bold green]")

        except TrainingError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1)

    app()
