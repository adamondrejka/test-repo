"""
Frame Extraction Pipeline Stage

Extracts frames from video and matches them to poses by timestamp.
"""

import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from tqdm import tqdm

console = Console()


@dataclass
class ExtractedFrame:
    """Represents an extracted frame with its matching pose."""
    frame_index: int
    timestamp: float
    image_path: Path
    pose_index: int
    pose_timestamp: float
    transform_matrix: List[float]
    tracking_state: str


class ExtractionError(Exception):
    """Error during frame extraction."""
    pass


def get_video_info(video_path: Path) -> Dict:
    """
    Get video metadata using ffprobe.

    Returns:
        Dict with: duration, fps, width, height, frame_count
    """
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams', '-show_format',
        str(video_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        raise ExtractionError(f"ffprobe failed: {e.stderr}")
    except json.JSONDecodeError as e:
        raise ExtractionError(f"Failed to parse ffprobe output: {e}")

    # Find video stream
    video_stream = None
    for stream in data.get('streams', []):
        if stream.get('codec_type') == 'video':
            video_stream = stream
            break

    if not video_stream:
        raise ExtractionError("No video stream found")

    # Parse frame rate
    fps_str = video_stream.get('r_frame_rate', '30/1')
    if '/' in fps_str:
        num, den = fps_str.split('/')
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    else:
        fps = float(fps_str)

    duration = float(data.get('format', {}).get('duration', 0))
    frame_count = int(video_stream.get('nb_frames', duration * fps))

    return {
        'duration': duration,
        'fps': fps,
        'width': int(video_stream.get('width', 0)),
        'height': int(video_stream.get('height', 0)),
        'frame_count': frame_count,
        'codec': video_stream.get('codec_name', 'unknown'),
    }


def extract_frames_ffmpeg(
    video_path: Path,
    output_dir: Path,
    fps: Optional[float] = None,
    quality: int = 2,
    start_number: int = 0
) -> int:
    """
    Extract frames from video using FFmpeg.

    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        fps: Output frame rate (None = use video fps)
        quality: JPEG quality (2=best, 31=worst)
        start_number: Starting frame number

    Returns:
        Number of frames extracted
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(output_dir / "frame_%06d.jpg")

    cmd = ['ffmpeg', '-y', '-i', str(video_path)]

    if fps:
        cmd.extend(['-vf', f'fps={fps}'])

    cmd.extend([
        '-qscale:v', str(quality),
        '-start_number', str(start_number),
        output_pattern
    ])

    console.print(f"[blue]Extracting frames from video...[/blue]")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise ExtractionError(f"FFmpeg failed: {e.stderr}")

    # Count extracted frames
    frames = list(output_dir.glob("frame_*.jpg"))
    console.print(f"[green]Extracted {len(frames)} frames[/green]")

    return len(frames)


def match_frames_to_poses(
    frame_dir: Path,
    poses: List[Dict],
    video_fps: float,
    video_start_time: float = 0.0,
    max_time_diff: float = 0.1
) -> List[ExtractedFrame]:
    """
    Match extracted frames to poses by timestamp.

    Args:
        frame_dir: Directory containing extracted frames
        poses: List of pose dictionaries from manifest
        video_fps: Video frame rate
        video_start_time: Timestamp of first video frame (matches first pose timestamp)
        max_time_diff: Maximum allowed time difference for matching (seconds)

    Returns:
        List of ExtractedFrame objects with matched poses
    """
    frames = sorted(frame_dir.glob("frame_*.jpg"))

    if not frames:
        raise ExtractionError("No frames found in directory")

    if not poses:
        raise ExtractionError("No poses provided")

    # Get pose timestamps
    pose_timestamps = np.array([p['timestamp'] for p in poses])

    # Determine video start time (align with first pose)
    if video_start_time == 0.0:
        video_start_time = pose_timestamps[0]

    matched = []
    unmatched_count = 0

    console.print(f"[blue]Matching {len(frames)} frames to {len(poses)} poses...[/blue]")

    for i, frame_path in enumerate(tqdm(frames, desc="Matching frames")):
        # Calculate frame timestamp
        frame_number = int(frame_path.stem.split('_')[1])
        frame_timestamp = video_start_time + (frame_number / video_fps)

        # Find nearest pose
        time_diffs = np.abs(pose_timestamps - frame_timestamp)
        nearest_idx = np.argmin(time_diffs)
        min_diff = time_diffs[nearest_idx]

        if min_diff <= max_time_diff:
            pose = poses[nearest_idx]
            matched.append(ExtractedFrame(
                frame_index=i,
                timestamp=frame_timestamp,
                image_path=frame_path,
                pose_index=nearest_idx,
                pose_timestamp=pose['timestamp'],
                transform_matrix=pose['transform_matrix'],
                tracking_state=pose.get('tracking_state', 'normal')
            ))
        else:
            unmatched_count += 1

    console.print(f"[green]Matched {len(matched)} frames[/green]")
    if unmatched_count > 0:
        console.print(f"[yellow]Skipped {unmatched_count} frames (no matching pose)[/yellow]")

    return matched


def filter_by_tracking_quality(
    frames: List[ExtractedFrame],
    include_limited: bool = True,
    include_lost: bool = False
) -> List[ExtractedFrame]:
    """
    Filter frames based on tracking quality.

    Args:
        frames: List of matched frames
        include_limited: Include frames with limited tracking
        include_lost: Include frames with lost tracking

    Returns:
        Filtered list of frames
    """
    filtered = []

    for frame in frames:
        if frame.tracking_state == 'normal':
            filtered.append(frame)
        elif frame.tracking_state == 'limited' and include_limited:
            filtered.append(frame)
        elif frame.tracking_state == 'not_available' and include_lost:
            filtered.append(frame)

    console.print(f"[blue]Filtered to {len(filtered)} frames with good tracking[/blue]")
    return filtered


def downsample_frames(
    frames: List[ExtractedFrame],
    target_count: Optional[int] = None,
    min_distance: Optional[float] = None
) -> List[ExtractedFrame]:
    """
    Downsample frames for training.

    Can either target a specific frame count or ensure minimum distance
    between consecutive frames.

    Args:
        frames: List of frames to downsample
        target_count: Target number of frames (takes priority)
        min_distance: Minimum distance in meters between frames

    Returns:
        Downsampled list of frames
    """
    if target_count and len(frames) <= target_count:
        return frames

    if target_count:
        # Uniform sampling
        indices = np.linspace(0, len(frames) - 1, target_count, dtype=int)
        return [frames[i] for i in indices]

    if min_distance:
        # Distance-based sampling
        from utils.matrix import row_major_to_matrix, extract_position

        selected = [frames[0]]
        last_pos = extract_position(row_major_to_matrix(frames[0].transform_matrix))

        for frame in frames[1:]:
            pos = extract_position(row_major_to_matrix(frame.transform_matrix))
            dist = np.linalg.norm(pos - last_pos)

            if dist >= min_distance:
                selected.append(frame)
                last_pos = pos

        return selected

    return frames


def save_frame_manifest(
    frames: List[ExtractedFrame],
    output_path: Path
) -> None:
    """
    Save frame-pose mapping to JSON for later use.
    """
    data = [
        {
            'frame_index': int(f.frame_index),
            'timestamp': float(f.timestamp),
            'image_path': str(f.image_path.name),
            'pose_index': int(f.pose_index),
            'pose_timestamp': float(f.pose_timestamp),
            'transform_matrix': [float(x) for x in f.transform_matrix],
            'tracking_state': f.tracking_state,
        }
        for f in frames
    ]

    with open(output_path, 'w') as fp:
        json.dump(data, fp, indent=2)

    console.print(f"[green]Saved frame manifest to {output_path}[/green]")


def extract_frames(
    video_path: Path,
    poses: List[Dict],
    output_dir: Path,
    target_fps: Optional[float] = None,
    include_limited_tracking: bool = True,
    target_frame_count: Optional[int] = None,
    min_frame_distance: Optional[float] = None
) -> Tuple[Path, List[ExtractedFrame]]:
    """
    Full frame extraction pipeline.

    Args:
        video_path: Path to input video
        poses: List of pose dictionaries from manifest
        output_dir: Output directory for frames
        target_fps: Target extraction FPS (None = video native)
        include_limited_tracking: Include limited tracking frames
        target_frame_count: Target number of output frames
        min_frame_distance: Minimum distance between frames (meters)

    Returns:
        Tuple of (frames_directory, list_of_matched_frames)
    """
    # Get video info
    video_info = get_video_info(video_path)
    console.print(f"[blue]Video: {video_info['width']}x{video_info['height']} @ "
                 f"{video_info['fps']:.1f}fps, {video_info['duration']:.1f}s[/blue]")

    # Create output directory
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames
    extraction_fps = target_fps or video_info['fps']
    num_frames = extract_frames_ffmpeg(
        video_path,
        frames_dir,
        fps=extraction_fps
    )

    # Match to poses
    matched = match_frames_to_poses(
        frames_dir,
        poses,
        video_fps=extraction_fps
    )

    # Filter by tracking quality
    matched = filter_by_tracking_quality(
        matched,
        include_limited=include_limited_tracking
    )

    # Downsample if requested
    if target_frame_count or min_frame_distance:
        matched = downsample_frames(
            matched,
            target_count=target_frame_count,
            min_distance=min_frame_distance
        )
        console.print(f"[blue]Downsampled to {len(matched)} frames[/blue]")

    # Save manifest
    save_frame_manifest(matched, output_dir / "frames_manifest.json")

    return frames_dir, matched


# CLI entry point
if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def main(
        work_dir: Path = typer.Argument(..., help="Work directory (from ingest stage)"),
        fps: Optional[float] = typer.Option(None, help="Target FPS for extraction"),
        target_frames: Optional[int] = typer.Option(None, help="Target number of frames"),
    ):
        """Extract frames from video and match to poses.

        Expects work_dir to contain a package subdirectory with:
        - scan_manifest.json
        - video.mov (or path from manifest)
        """
        # Find the package directory (created by ingest)
        manifest_candidates = list(work_dir.rglob("scan_manifest.json"))
        if not manifest_candidates:
            console.print(f"[bold red]Error:[/bold red] No scan_manifest.json found in {work_dir}")
            raise typer.Exit(1)

        manifest_path = manifest_candidates[0]
        package_dir = manifest_path.parent

        # Load manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        poses = manifest['poses']
        video_path = package_dir / manifest['assets']['video_path']

        if not video_path.exists():
            console.print(f"[bold red]Error:[/bold red] Video not found: {video_path}")
            raise typer.Exit(1)

        try:
            frames_dir, matched = extract_frames(
                video_path,
                poses,
                work_dir / "extracted",
                target_fps=fps,
                target_frame_count=target_frames
            )
            console.print(f"\n[bold green]Extraction complete![/bold green]")
            console.print(f"Frames: {frames_dir}")
            console.print(f"Matched: {len(matched)}")
        except ExtractionError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1)

    @app.command("from-paths")
    def from_paths(
        video_path: Path = typer.Argument(..., help="Path to video file"),
        poses_path: Path = typer.Argument(..., help="Path to poses JSON or manifest"),
        output_dir: Path = typer.Option(Path("./output"), help="Output directory"),
        fps: Optional[float] = typer.Option(None, help="Target FPS for extraction"),
        target_frames: Optional[int] = typer.Option(None, help="Target number of frames"),
    ):
        """Extract frames from explicit video and poses paths."""
        # Load poses
        with open(poses_path, 'r') as f:
            data = json.load(f)

        if 'poses' in data:
            poses = data['poses']
        else:
            poses = data

        try:
            frames_dir, matched = extract_frames(
                video_path,
                poses,
                output_dir,
                target_fps=fps,
                target_frame_count=target_frames
            )
            console.print(f"\n[bold green]Extraction complete![/bold green]")
            console.print(f"Frames: {frames_dir}")
            console.print(f"Matched: {len(matched)}")
        except ExtractionError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1)

    app()
