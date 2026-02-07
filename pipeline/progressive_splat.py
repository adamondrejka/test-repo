"""
Progressive Multi-Resolution Splat (LOD) Generation

Splits a cleaned Gaussian Splat PLY into multiple levels of detail
sorted by visual importance for progressive loading. LOD0 loads first
(3-5s to first render), then LOD1 and LOD2 stream in.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from rich.console import Console

from .compress import read_gaussian_ply, compress_ply_to_spz

console = Console()


# LOD split ratios: LOD0 = top 15%, LOD1 = next 35%, LOD2 = remaining 50%
LOD_SPLITS = [0.15, 0.35, 0.50]


def compute_importance(opacities: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    Compute visual importance score for each gaussian.

    Score = sigmoid(opacity) * exp(max(scale_dims))

    High-opacity, large gaussians are most important for the coarse view.
    """
    # Sigmoid of raw opacity logit
    sigmoid_opacity = 1.0 / (1.0 + np.exp(-np.clip(opacities, -10, 10)))
    # Max scale across 3 dimensions (log-space values from training)
    max_scale = scales.max(axis=1)
    # exp(max_scale) gives world-space size
    size = np.exp(np.clip(max_scale, -10, 10))
    return sigmoid_opacity * size


def write_gaussian_ply(
    path: Path,
    positions: np.ndarray,
    colors: Optional[np.ndarray],
    scales: Optional[np.ndarray],
    rotations: Optional[np.ndarray],
    opacities: Optional[np.ndarray],
    sh_coeffs: Optional[np.ndarray],
) -> int:
    """
    Write a subset of gaussians to a PLY file.

    Returns the number of gaussians written.
    """
    n = len(positions)

    # Build property list and data columns
    props = []
    columns = []

    # Positions (always present)
    for i, name in enumerate(['x', 'y', 'z']):
        props.append(f"property float {name}")
        columns.append(positions[:, i])

    # SH DC coefficients (colors)
    if colors is not None:
        for i, name in enumerate(['f_dc_0', 'f_dc_1', 'f_dc_2']):
            props.append(f"property float {name}")
            columns.append(colors[:, i])

    # SH rest coefficients
    if sh_coeffs is not None:
        for i in range(sh_coeffs.shape[1]):
            props.append(f"property float f_rest_{i}")
            columns.append(sh_coeffs[:, i])

    # Opacity
    if opacities is not None:
        props.append("property float opacity")
        columns.append(opacities)

    # Scales
    if scales is not None:
        for i, name in enumerate(['scale_0', 'scale_1', 'scale_2']):
            props.append(f"property float {name}")
            columns.append(scales[:, i])

    # Rotations
    if rotations is not None:
        for i, name in enumerate(['rot_0', 'rot_1', 'rot_2', 'rot_3']):
            props.append(f"property float {name}")
            columns.append(rotations[:, i])

    # Build header
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "\n".join(props) + "\n"
        "end_header\n"
    )

    # Stack all columns into structured array
    dtype = [(f"col_{i}", np.float32) for i in range(len(columns))]
    data = np.empty(n, dtype=dtype)
    for i, col in enumerate(columns):
        data[f"col_{i}"] = col.astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(header.encode('utf-8'))
        f.write(data.tobytes())

    return n


def generate_progressive_lods(
    ply_path: Path,
    output_dir: Path,
    compress_spz: bool = True,
    compression_quality: str = "high",
) -> Tuple[List[Dict], Dict]:
    """
    Split a Gaussian Splat PLY into 3 LOD levels by visual importance.

    Args:
        ply_path: Path to cleaned PLY file
        output_dir: Directory to write LOD files
        compress_spz: Whether to also compress each LOD to SPZ
        compression_quality: SPZ quality level

    Returns:
        Tuple of (lod_levels list, stats dict)
    """
    console.print("[blue]Generating progressive LOD levels...[/blue]")

    # Read all gaussian data
    positions, colors, scales, rotations, opacities, sh_coeffs = read_gaussian_ply(ply_path)
    total = len(positions)
    console.print(f"  Total gaussians: {total:,}")

    if opacities is None or scales is None:
        console.print("[yellow]Missing opacity/scale data, skipping LOD generation[/yellow]")
        return [], {}

    # Compute importance and sort
    importance = compute_importance(opacities, scales)
    sorted_indices = np.argsort(importance)[::-1]  # descending

    # Split into LOD levels
    lod_levels = []
    offset = 0
    lod_names = ["scene_lod0", "scene_lod1", "scene_lod2"]

    for level, (name, ratio) in enumerate(zip(lod_names, LOD_SPLITS)):
        count = int(total * ratio)
        # Last level gets any rounding remainder
        if level == len(LOD_SPLITS) - 1:
            count = total - offset
        indices = sorted_indices[offset:offset + count]
        offset += count

        # Slice data for this LOD
        lod_pos = positions[indices]
        lod_colors = colors[indices] if colors is not None else None
        lod_scales = scales[indices]
        lod_rots = rotations[indices] if rotations is not None else None
        lod_opac = opacities[indices]
        lod_sh = sh_coeffs[indices] if sh_coeffs is not None else None

        # Write PLY
        ply_out = output_dir / f"{name}.ply"
        n_written = write_gaussian_ply(
            ply_out, lod_pos, lod_colors, lod_scales, lod_rots, lod_opac, lod_sh)
        ply_size_mb = ply_out.stat().st_size / (1024 * 1024)

        lod_info = {
            "level": level,
            "file": f"{name}.ply",
            "gaussians": n_written,
            "size_mb": round(ply_size_mb, 1),
        }

        # Optionally compress to SPZ
        if compress_spz:
            try:
                spz_out = output_dir / f"{name}.spz"
                spz_out, _ = compress_ply_to_spz(ply_out, spz_out, quality=compression_quality)
                spz_size_mb = spz_out.stat().st_size / (1024 * 1024)
                lod_info["file_spz"] = f"{name}.spz"
                lod_info["size_spz_mb"] = round(spz_size_mb, 1)
            except Exception as e:
                console.print(f"[yellow]SPZ compression failed for {name}: {e}[/yellow]")

        lod_levels.append(lod_info)
        console.print(f"  LOD{level}: {n_written:,} gaussians ({ply_size_mb:.1f} MB)")

    stats = {
        "total_gaussians": total,
        "lod_count": len(lod_levels),
        "split_ratios": LOD_SPLITS,
    }

    console.print(f"[green]Progressive LOD generation complete ({len(lod_levels)} levels)[/green]")
    return lod_levels, stats
