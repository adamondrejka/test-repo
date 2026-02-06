"""
Compression Pipeline Stage

Compresses Gaussian Splat PLY files to SPZ format using Niantic's compression.
"""

import json
import struct
import subprocess
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
from rich.console import Console

console = Console()


class CompressionError(Exception):
    """Error during compression."""
    pass


def get_ply_info(ply_path: Path) -> Dict:
    """
    Get information about a PLY file.

    Returns:
        Dict with point_count, file_size, properties
    """
    if not ply_path.exists():
        raise CompressionError(f"PLY file not found: {ply_path}")

    info = {
        'file_size': ply_path.stat().st_size,
        'file_size_mb': ply_path.stat().st_size / (1024 * 1024),
    }

    # Parse PLY header
    with open(ply_path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
            if len(header_lines) > 100:  # Safety limit
                break

    # Extract info from header
    properties = []
    for line in header_lines:
        if line.startswith('element vertex'):
            info['point_count'] = int(line.split()[-1])
        elif line.startswith('property'):
            parts = line.split()
            if len(parts) >= 3:
                properties.append({
                    'type': parts[1],
                    'name': parts[2],
                })

    info['properties'] = properties
    info['property_count'] = len(properties)

    return info


def compress_ply_to_spz(
    ply_path: Path,
    output_path: Path,
    quality: str = "high"
) -> Tuple[Path, Dict]:
    """
    Compress PLY to SPZ format.

    This uses Niantic's SPZ compression format which provides ~90% size reduction
    while maintaining visual quality.

    Note: SPZ compression requires the spz tool to be installed.
    If not available, falls back to a simple quantization approach.

    Args:
        ply_path: Path to input PLY file
        output_path: Path for output SPZ file
        quality: Compression quality ("low", "medium", "high")

    Returns:
        Tuple of (output_path, compression_stats)
    """
    if not ply_path.exists():
        raise CompressionError(f"Input PLY not found: {ply_path}")

    output_path = output_path.with_suffix('.spz')
    input_info = get_ply_info(ply_path)

    console.print(f"[blue]Compressing {ply_path.name}...[/blue]")
    console.print(f"  Input size: {input_info['file_size_mb']:.1f} MB")
    console.print(f"  Points: {input_info.get('point_count', 'unknown')}")

    # Try using spz tool if available
    try:
        result = subprocess.run(
            ['spz', '--version'],
            capture_output=True,
            text=True
        )
        spz_available = result.returncode == 0
    except FileNotFoundError:
        spz_available = False

    if spz_available:
        # Use official SPZ compression
        compress_with_spz_tool(ply_path, output_path, quality)
    else:
        # Fallback: Create a minimal SPZ-like compressed format
        console.print("[yellow]SPZ tool not found, using fallback compression[/yellow]")
        compress_ply_fallback(ply_path, output_path)

    # Calculate compression stats
    output_size = output_path.stat().st_size
    compression_ratio = input_info['file_size'] / output_size if output_size > 0 else 1

    stats = {
        'input_size': input_info['file_size'],
        'output_size': output_size,
        'input_size_mb': input_info['file_size_mb'],
        'output_size_mb': output_size / (1024 * 1024),
        'compression_ratio': compression_ratio,
        'size_reduction_percent': (1 - output_size / input_info['file_size']) * 100,
    }

    console.print(f"[green]Compression complete![/green]")
    console.print(f"  Output size: {stats['output_size_mb']:.1f} MB")
    console.print(f"  Ratio: {stats['compression_ratio']:.1f}x "
                 f"({stats['size_reduction_percent']:.1f}% reduction)")

    return output_path, stats


def compress_with_spz_tool(
    ply_path: Path,
    output_path: Path,
    quality: str
) -> None:
    """Use the official SPZ compression tool."""
    quality_map = {
        'low': '0',
        'medium': '1',
        'high': '2',
    }

    cmd = [
        'spz', 'compress',
        '-i', str(ply_path),
        '-o', str(output_path),
        '-q', quality_map.get(quality, '2'),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise CompressionError(f"SPZ compression failed: {e.stderr}")


def compress_ply_fallback(ply_path: Path, output_path: Path) -> None:
    """
    Fallback compression when SPZ tool is not available.

    Creates a simplified compressed format that can still be loaded
    by compatible viewers.
    """
    # Read PLY data
    points, colors, scales, rotations, opacities, sh_coeffs = read_gaussian_ply(ply_path)

    # Quantize and compress
    compressed_data = {
        'version': 1,
        'format': 'spz_fallback',
        'point_count': len(points),
    }

    # Quantize positions to 16-bit
    pos_min = points.min(axis=0)
    pos_max = points.max(axis=0)
    pos_range = pos_max - pos_min
    pos_range[pos_range == 0] = 1  # Avoid division by zero

    positions_normalized = (points - pos_min) / pos_range
    positions_quantized = (positions_normalized * 65535).astype(np.uint16)

    compressed_data['bounds'] = {
        'min': pos_min.tolist(),
        'max': pos_max.tolist(),
    }

    # Write binary data
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        # Write header
        header = json.dumps(compressed_data).encode('utf-8')
        f.write(struct.pack('<I', len(header)))
        f.write(header)

        # Write quantized positions
        f.write(positions_quantized.tobytes())

        # Write colors (8-bit per channel)
        if colors is not None:
            colors_u8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
            f.write(colors_u8.tobytes())

        # Write scales (16-bit quantized)
        if scales is not None:
            scales_normalized = (scales - scales.min()) / (scales.max() - scales.min() + 1e-6)
            scales_u16 = (scales_normalized * 65535).astype(np.uint16)
            f.write(scales_u16.tobytes())

        # Write rotations (8-bit quaternion)
        if rotations is not None:
            rotations_i8 = (np.clip(rotations, -1, 1) * 127).astype(np.int8)
            f.write(rotations_i8.tobytes())

        # Write opacities (8-bit)
        if opacities is not None:
            opacity_u8 = (np.clip(opacities, 0, 1) * 255).astype(np.uint8)
            f.write(opacity_u8.tobytes())


def read_gaussian_ply(ply_path: Path) -> Tuple:
    """
    Read Gaussian Splat PLY file.

    Returns:
        Tuple of (positions, colors, scales, rotations, opacities, sh_coeffs)
    """
    with open(ply_path, 'rb') as f:
        # Read header
        header = ""
        while True:
            line = f.readline().decode('utf-8', errors='ignore')
            header += line
            if 'end_header' in line:
                break

        # Parse header
        vertex_count = 0
        properties = []
        for line in header.split('\n'):
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                if len(parts) >= 3:
                    properties.append((parts[1], parts[2]))

        # Create dtype for reading
        type_map = {
            'float': np.float32,
            'double': np.float64,
            'uchar': np.uint8,
            'int': np.int32,
        }

        dtype = []
        for ptype, pname in properties:
            np_type = type_map.get(ptype, np.float32)
            dtype.append((pname, np_type))

        # Read binary data
        data = np.frombuffer(f.read(), dtype=np.dtype(dtype))

    # Extract components
    positions = np.column_stack([data['x'], data['y'], data['z']])

    # Colors (if present)
    colors = None
    if 'f_dc_0' in data.dtype.names:
        colors = np.column_stack([
            data['f_dc_0'], data['f_dc_1'], data['f_dc_2']
        ])

    # Scales
    scales = None
    if 'scale_0' in data.dtype.names:
        scales = np.column_stack([
            data['scale_0'], data['scale_1'], data['scale_2']
        ])

    # Rotations
    rotations = None
    if 'rot_0' in data.dtype.names:
        rotations = np.column_stack([
            data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3']
        ])

    # Opacities
    opacities = None
    if 'opacity' in data.dtype.names:
        opacities = data['opacity']

    # Spherical harmonics
    sh_coeffs = None
    sh_names = [n for n in data.dtype.names if n.startswith('f_rest_')]
    if sh_names:
        sh_coeffs = np.column_stack([data[n] for n in sorted(sh_names)])

    return positions, colors, scales, rotations, opacities, sh_coeffs


def validate_spz(spz_path: Path) -> bool:
    """Validate an SPZ file can be read."""
    if not spz_path.exists():
        return False

    try:
        with open(spz_path, 'rb') as f:
            header_len = struct.unpack('<I', f.read(4))[0]
            header = json.loads(f.read(header_len).decode('utf-8'))
            return 'point_count' in header or 'version' in header
    except Exception:
        return False


# CLI entry point
if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def main(
        input_path: Path = typer.Argument(..., help="Input PLY file"),
        output_path: Optional[Path] = typer.Option(None, help="Output SPZ file"),
        quality: str = typer.Option("high", help="Quality: low, medium, high"),
    ):
        """Compress Gaussian Splat PLY to SPZ format."""
        if output_path is None:
            output_path = input_path.with_suffix('.spz')

        try:
            out_path, stats = compress_ply_to_spz(input_path, output_path, quality)
            console.print(f"\n[bold green]Compressed to {out_path}[/bold green]")
        except CompressionError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1)

    @app.command("info")
    def info(ply_path: Path = typer.Argument(..., help="PLY file to inspect")):
        """Show PLY file information."""
        try:
            info = get_ply_info(ply_path)
            console.print(f"[bold]PLY Info: {ply_path.name}[/bold]")
            console.print(f"  Size: {info['file_size_mb']:.2f} MB")
            console.print(f"  Points: {info.get('point_count', 'unknown')}")
            console.print(f"  Properties: {info.get('property_count', 0)}")
        except CompressionError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1)

    app()
