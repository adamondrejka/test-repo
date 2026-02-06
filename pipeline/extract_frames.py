"""
Frame Extraction Pipeline Stage

Extracts frames from video and matches them to poses by timestamp.
Smart frame selection based on camera movement for optimal training coverage.
"""

import json
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from tqdm import tqdm

console = Console()

# Default target frame count for training
# 250 frames works well for most room-sized scans
# Increase for larger spaces (warehouses, outdoor)
DEFAULT_TARGET_FRAMES = 250

# Minimum camera movement (meters) between frames as fallback
DEFAULT_MIN_DISTANCE = 0.05  # 5cm

# Threshold for detecting zero/invalid positions (meters)
# ARKit returns near-zero positions when tracking is unstable
# 5cm threshold catches all "origin cluster" poses
ZERO_POSITION_THRESHOLD = 0.05


def is_valid_pose(pose: dict) -> bool:
    """
    Check if pose has valid position (not at origin).

    ARKit returns [0, 0, 0] position when tracking is lost.
    These poses corrupt the Gaussian Splat reconstruction.

    Args:
        pose: Pose dictionary with transform_matrix (16 elements, row-major)

    Returns:
        True if position is valid (not at origin)
    """
    tm = pose.get('transform_matrix', [])
    if len(tm) != 16:
        return False
    # Position is at indices 3, 7, 11 in row-major 4x4 matrix
    # [r00 r01 r02 tx]   -> indices 0-3
    # [r10 r11 r12 ty]   -> indices 4-7
    # [r20 r21 r22 tz]   -> indices 8-11
    # [0   0   0   1 ]   -> indices 12-15
    pos = [tm[3], tm[7], tm[11]]
    # Reject if position is essentially zero (tracking lost)
    return not (abs(pos[0]) < ZERO_POSITION_THRESHOLD and
                abs(pos[1]) < ZERO_POSITION_THRESHOLD and
                abs(pos[2]) < ZERO_POSITION_THRESHOLD)


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
    max_time_diff: float = 0.1,
    include_limited: bool = True
) -> List[ExtractedFrame]:
    """
    Match extracted frames to poses by timestamp with position validation.

    Rejects poses with zero positions (ARKit tracking lost) during matching
    to prevent corrupted data from entering the training pipeline.

    Args:
        frame_dir: Directory containing extracted frames
        poses: List of pose dictionaries from manifest
        video_fps: Video frame rate
        video_start_time: Timestamp of first video frame (matches first pose timestamp)
        max_time_diff: Maximum allowed time difference for matching (seconds)
        include_limited: Include frames with 'limited' tracking state

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
    zero_position_count = 0
    limited_tracking_count = 0

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

            # Validate pose has non-zero position (tracking was active)
            if not is_valid_pose(pose):
                zero_position_count += 1
                continue

            # Check tracking state
            tracking_state = pose.get('tracking_state', 'normal')
            if tracking_state == 'limited' and not include_limited:
                limited_tracking_count += 1
                continue

            matched.append(ExtractedFrame(
                frame_index=i,
                timestamp=frame_timestamp,
                image_path=frame_path,
                pose_index=nearest_idx,
                pose_timestamp=pose['timestamp'],
                transform_matrix=pose['transform_matrix'],
                tracking_state=tracking_state
            ))
        else:
            unmatched_count += 1

    console.print(f"[green]Matched {len(matched)} frames[/green]")
    if unmatched_count > 0:
        console.print(f"[yellow]Skipped {unmatched_count} frames (no matching pose)[/yellow]")
    if zero_position_count > 0:
        console.print(f"[yellow]Skipped {zero_position_count} frames (zero position - tracking lost)[/yellow]")
    if limited_tracking_count > 0:
        console.print(f"[yellow]Skipped {limited_tracking_count} frames (limited tracking)[/yellow]")

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
    min_distance: Optional[float] = None,
    use_movement_based: bool = True
) -> List[ExtractedFrame]:
    """
    Downsample frames for training using smart selection.

    Selection strategy (in order of priority):
    1. If target_count specified: select frames with maximum coverage
    2. If min_distance specified: ensure minimum camera movement between frames
    3. If use_movement_based: distribute frames based on camera path length

    Args:
        frames: List of frames to downsample
        target_count: Target number of frames
        min_distance: Minimum distance in meters between frames
        use_movement_based: Use movement-based selection (better coverage)

    Returns:
        Downsampled list of frames
    """
    if len(frames) == 0:
        return frames

    if target_count and len(frames) <= target_count:
        return frames

    from utils.matrix import row_major_to_matrix

    def extract_position(matrix):
        """Extract camera position from 4x4 transform matrix."""
        return np.array(matrix[:3, 3])

    # Calculate positions for all frames
    positions = []
    for frame in frames:
        matrix = row_major_to_matrix(frame.transform_matrix)
        positions.append(extract_position(matrix))
    positions = np.array(positions)

    if target_count and use_movement_based:
        # Movement-based selection: select frames distributed along camera path
        # Calculate cumulative distance along path
        distances = [0.0]
        for i in range(1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[i-1])
            distances.append(distances[-1] + dist)

        total_distance = distances[-1]
        if total_distance < 0.01:  # Camera didn't move much
            # Fall back to uniform temporal sampling
            indices = np.linspace(0, len(frames) - 1, target_count, dtype=int)
            return [frames[i] for i in indices]

        # Select frames at uniform distances along path
        target_distances = np.linspace(0, total_distance, target_count)
        selected_indices = []

        for target_dist in target_distances:
            # Find frame closest to this distance along path
            idx = np.argmin(np.abs(np.array(distances) - target_dist))
            if idx not in selected_indices:
                selected_indices.append(idx)
            else:
                # Find nearest unselected frame
                for offset in range(1, len(frames)):
                    for candidate in [idx + offset, idx - offset]:
                        if 0 <= candidate < len(frames) and candidate not in selected_indices:
                            selected_indices.append(candidate)
                            break
                    else:
                        continue
                    break

        selected_indices = sorted(set(selected_indices))
        console.print(f"[blue]Movement-based selection: {len(selected_indices)} frames "
                     f"covering {total_distance:.2f}m path[/blue]")
        return [frames[i] for i in selected_indices]

    if target_count:
        # Uniform temporal sampling (fallback)
        indices = np.linspace(0, len(frames) - 1, target_count, dtype=int)
        return [frames[i] for i in indices]

    if min_distance:
        # Distance-based sampling - select frame when camera moves enough
        selected = [frames[0]]
        last_pos = positions[0]

        for i, frame in enumerate(frames[1:], 1):
            dist = np.linalg.norm(positions[i] - last_pos)
            if dist >= min_distance:
                selected.append(frame)
                last_pos = positions[i]

        # Always include last frame for full coverage
        if selected[-1] != frames[-1]:
            selected.append(frames[-1])

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


def cleanup_unused_frames(
    frames_dir: Path,
    keep_frames: List[ExtractedFrame]
) -> int:
    """
    Remove frames that weren't selected to save disk space.

    Args:
        frames_dir: Directory containing all extracted frames
        keep_frames: List of frames to keep

    Returns:
        Number of frames removed
    """
    keep_paths = {f.image_path.name for f in keep_frames}
    all_frames = list(frames_dir.glob("frame_*.jpg")) + list(frames_dir.glob("frame_*.png"))

    removed = 0
    for frame_path in all_frames:
        if frame_path.name not in keep_paths:
            frame_path.unlink()
            removed += 1

    if removed > 0:
        console.print(f"[dim]Cleaned up {removed} unused frames[/dim]")

    return removed


def extract_frames(
    video_path: Path,
    poses: List[Dict],
    output_dir: Path,
    target_fps: Optional[float] = None,
    include_limited_tracking: bool = True,
    target_frame_count: Optional[int] = DEFAULT_TARGET_FRAMES,
    min_frame_distance: Optional[float] = None,
    cleanup_unused: bool = True,
    video_start_time: Optional[float] = 0.0
) -> Tuple[Path, List[ExtractedFrame]]:
    """
    Full frame extraction pipeline with smart frame selection.

    By default, extracts ~250 frames using movement-based selection
    to ensure good coverage of the scanned space.

    Args:
        video_path: Path to input video
        poses: List of pose dictionaries from manifest
        output_dir: Output directory for frames
        target_fps: Target extraction FPS (None = video native)
        include_limited_tracking: Include limited tracking frames
        target_frame_count: Target number of output frames (default: 250)
                           Set to None to disable downsampling
        min_frame_distance: Minimum distance between frames (meters)
        cleanup_unused: Remove non-selected frames to save disk space
        video_start_time: System uptime of the first frame (for synchronization)

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

    # Match to poses (includes position and tracking validation)
    matched = match_frames_to_poses(
        frames_dir,
        poses,
        video_fps=extraction_fps,
        video_start_time=video_start_time or 0.0,
        include_limited=include_limited_tracking
    )

    # Smart downsampling (enabled by default)
    original_count = len(matched)
    if target_frame_count or min_frame_distance:
        matched = downsample_frames(
            matched,
            target_count=target_frame_count,
            min_distance=min_frame_distance,
            use_movement_based=True
        )
        console.print(f"[green]Selected {len(matched)}/{original_count} frames "
                     f"(target: {target_frame_count or 'distance-based'})[/green]")

    # Clean up unused frames to save disk space
    if cleanup_unused and len(matched) < original_count:
        cleanup_unused_frames(frames_dir, matched)

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
        target_frames: int = typer.Option(DEFAULT_TARGET_FRAMES, help="Target number of frames (default: 250)"),
        no_limit: bool = typer.Option(False, "--no-limit", help="Disable frame limit (extract all)"),
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
        video_start_time = manifest.get('video_start_time')

        if not video_path.exists():
            console.print(f"[bold red]Error:[/bold red] Video not found: {video_path}")
            raise typer.Exit(1)

        try:
            frames_dir, matched = extract_frames(
                video_path,
                poses,
                work_dir / "extracted",
                target_fps=fps,
                target_frame_count=None if no_limit else target_frames,
                video_start_time=video_start_time
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
