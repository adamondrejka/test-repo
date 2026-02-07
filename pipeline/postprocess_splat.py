"""
Post-Training Gaussian Splat Cleanup

Removes floaters, culls near-transparent splats, clamps extreme scales,
and crops splats outside the camera trajectory bounding box.
"""

import struct
from pathlib import Path
from typing import Optional, Dict
import numpy as np
from rich.console import Console

from .compress import read_gaussian_ply

console = Console()


class PostprocessError(Exception):
    """Error during postprocessing."""
    pass


def cleanup_splat(
    ply_path: Path,
    output_path: Path,
    camera_positions: Optional[np.ndarray] = None,
    min_opacity: float = 0.05,
    floater_neighbors: int = 5,
    floater_radius: float = 0.08,
    max_scale: float = 0.8,
    bbox_margin: float = 1.5,
) -> Dict:
    """
    Clean up a Gaussian Splat PLY file.

    Applies in order:
    1. Remove near-transparent splats (opacity < min_opacity)
    2. Clamp extreme scales
    3. Crop outside camera trajectory bounding box (if positions provided)
    4. Remove floaters (statistical outlier removal)

    Args:
        ply_path: Input PLY file
        output_path: Output cleaned PLY file
        camera_positions: Nx3 array of camera positions for bbox cropping
        min_opacity: Minimum opacity threshold
        floater_neighbors: Minimum neighbors within radius to keep
        floater_radius: Radius for neighbor search (meters)
        max_scale: Maximum allowed scale value (in log space from PLY)
        bbox_margin: Margin around camera bbox for cropping (meters)

    Returns:
        Dict with cleanup statistics
    """
    console.print(f"[blue]Loading splat from {ply_path.name}...[/blue]")
    positions, colors, scales, rotations, opacities, sh_coeffs = read_gaussian_ply(ply_path)

    initial_count = len(positions)
    mask = np.ones(initial_count, dtype=bool)
    stats = {"initial_count": initial_count}

    # 1. Cull near-transparent splats
    if opacities is not None:
        # Nerfstudio stores opacity in logit space, convert to sigmoid
        sigmoid_opacity = 1.0 / (1.0 + np.exp(-opacities))
        opacity_mask = sigmoid_opacity >= min_opacity
        removed = (~opacity_mask).sum()
        mask &= opacity_mask
        stats["transparent_removed"] = int(removed)
        if removed > 0:
            console.print(f"[dim]  Removed {removed} transparent splats "
                          f"(opacity < {min_opacity})[/dim]")

    # 2. Clamp extreme scales
    if scales is not None:
        # Scales are in log space, exp(scale) is the actual size
        scale_max = np.max(scales, axis=1)
        scale_mask = scale_max <= max_scale
        removed = mask.sum() - (mask & scale_mask).sum()
        mask &= scale_mask
        stats["oversized_removed"] = int(removed)
        if removed > 0:
            console.print(f"[dim]  Removed {removed} oversized splats "
                          f"(log_scale > {max_scale})[/dim]")

    # 3. Crop outside camera bounding box
    if camera_positions is not None and len(camera_positions) > 0:
        cam_min = camera_positions.min(axis=0) - bbox_margin
        cam_max = camera_positions.max(axis=0) + bbox_margin
        in_bbox = np.all(
            (positions >= cam_min) & (positions <= cam_max), axis=1
        )
        removed = mask.sum() - (mask & in_bbox).sum()
        mask &= in_bbox
        stats["out_of_bounds_removed"] = int(removed)
        if removed > 0:
            console.print(f"[dim]  Removed {removed} out-of-bounds splats[/dim]")

    # 4. Statistical outlier removal (floaters)
    # Use a grid-based approach for efficiency instead of KD-tree
    remaining_positions = positions[mask]
    remaining_indices = np.where(mask)[0]

    if len(remaining_positions) > 100:
        # Voxel grid for fast neighbor counting
        voxel_size = floater_radius
        voxel_coords = np.floor(remaining_positions / voxel_size).astype(int)

        # Count points per voxel
        from collections import Counter
        voxel_keys = [tuple(v) for v in voxel_coords]
        voxel_counts = Counter(voxel_keys)

        # Count neighbors: sum of adjacent 3x3x3 voxel counts
        floater_mask = np.zeros(len(remaining_positions), dtype=bool)
        for i, vk in enumerate(voxel_keys):
            neighbor_count = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor_count += voxel_counts.get(
                            (vk[0] + dx, vk[1] + dy, vk[2] + dz), 0)
            # Subtract self
            neighbor_count -= 1
            if neighbor_count < floater_neighbors:
                floater_mask[i] = True

        removed = floater_mask.sum()
        # Apply floater mask back to the main mask
        for i, is_floater in enumerate(floater_mask):
            if is_floater:
                mask[remaining_indices[i]] = False

        stats["floaters_removed"] = int(removed)
        if removed > 0:
            console.print(f"[dim]  Removed {removed} floater splats "
                          f"(<{floater_neighbors} neighbors)[/dim]")

    final_count = mask.sum()
    stats["final_count"] = int(final_count)
    stats["total_removed"] = initial_count - int(final_count)
    stats["removal_percent"] = round(
        (1 - final_count / initial_count) * 100, 1) if initial_count > 0 else 0

    console.print(f"[green]Cleanup: {initial_count} → {final_count} splats "
                  f"({stats['removal_percent']}% removed)[/green]")

    # Write cleaned PLY
    _write_filtered_ply(ply_path, output_path, mask)

    return stats


def _write_filtered_ply(
    input_path: Path,
    output_path: Path,
    mask: np.ndarray,
) -> None:
    """Write a filtered PLY by copying header and selected vertices."""
    with open(input_path, 'rb') as f:
        # Read header
        header_bytes = b""
        while True:
            line = f.readline()
            header_bytes += line
            if b'end_header' in line:
                break

        # Parse vertex count from header and compute record size
        header_text = header_bytes.decode('utf-8', errors='ignore')
        vertex_count = 0
        properties = []
        for line in header_text.split('\n'):
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                ptype = parts[1]
                properties.append(ptype)

        type_sizes = {'float': 4, 'double': 8, 'uchar': 1, 'int': 4}
        record_size = sum(type_sizes.get(p, 4) for p in properties)

        # Read all vertex data
        all_data = f.read(vertex_count * record_size)

    # Update header with new count
    new_count = int(mask.sum())
    new_header = header_bytes.decode('utf-8', errors='ignore')
    new_header = new_header.replace(
        f'element vertex {vertex_count}',
        f'element vertex {new_count}'
    )

    # Write filtered PLY
    with open(output_path, 'wb') as f:
        f.write(new_header.encode('utf-8'))
        for i in range(vertex_count):
            if mask[i]:
                start = i * record_size
                f.write(all_data[start:start + record_size])

    size_mb = output_path.stat().st_size / (1024 * 1024)
    console.print(f"[green]Wrote cleaned splat: {output_path.name} ({size_mb:.1f} MB)[/green]")
