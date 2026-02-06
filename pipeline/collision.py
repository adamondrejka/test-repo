"""
Collision Mesh Pipeline Stage

Generates collision meshes and floorplans from RoomPlan USDZ data.
"""

import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
from rich.console import Console

console = Console()


class CollisionError(Exception):
    """Error during collision mesh generation."""
    pass


def parse_usdz_to_mesh(usdz_path: Path) -> Dict:
    """
    Parse USDZ file to extract mesh geometry.

    Note: Full USDZ parsing requires the pxr (USD) library.
    This provides a fallback using usdcat if available.

    Returns:
        Dict with vertices, faces, and metadata
    """
    if not usdz_path.exists():
        raise CollisionError(f"USDZ file not found: {usdz_path}")

    # Try using USD tools if available
    try:
        result = subprocess.run(
            ['usdcat', '--help'],
            capture_output=True,
            text=True
        )
        usd_available = result.returncode == 0
    except FileNotFoundError:
        usd_available = False

    if usd_available:
        return parse_usdz_with_usdcat(usdz_path)
    else:
        console.print("[yellow]USD tools not found. Using trimesh for USDZ parsing.[/yellow]")
        return parse_usdz_with_trimesh(usdz_path)


def parse_usdz_with_usdcat(usdz_path: Path) -> Dict:
    """Parse USDZ using USD command-line tools."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract USDZ
        usda_path = Path(tmpdir) / "scene.usda"

        result = subprocess.run(
            ['usdcat', '-o', str(usda_path), '--flatten', str(usdz_path)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise CollisionError(f"Failed to parse USDZ: {result.stderr}")

        # Parse USDA text format
        with open(usda_path, 'r') as f:
            usda_content = f.read()

        return parse_usda_content(usda_content)


def parse_usda_content(content: str) -> Dict:
    """Parse USDA text content to extract mesh data."""
    vertices = []
    faces = []

    # Simple parser for USDA point/face arrays
    lines = content.split('\n')
    in_points = False
    in_faces = False
    current_array = []

    for line in lines:
        line = line.strip()

        if 'point3f[] points' in line:
            in_points = True
            current_array = []
        elif 'int[] faceVertexIndices' in line:
            in_faces = True
            current_array = []
        elif in_points and line.startswith(']'):
            vertices = current_array
            in_points = False
        elif in_faces and line.startswith(']'):
            faces = current_array
            in_faces = False
        elif in_points and line.startswith('('):
            # Parse point: (x, y, z)
            coords = line.strip('(),').split(',')
            if len(coords) >= 3:
                vertices.append([float(c.strip()) for c in coords[:3]])
        elif in_faces and line:
            # Parse face indices
            indices = [int(x) for x in line.strip(',').split(',') if x.strip().isdigit()]
            faces.extend(indices)

    return {
        'vertices': np.array(vertices) if vertices else np.array([]),
        'faces': np.array(faces) if faces else np.array([]),
    }


def parse_usdz_with_trimesh(usdz_path: Path) -> Dict:
    """Parse USDZ using trimesh library."""
    try:
        import trimesh
    except ImportError:
        raise CollisionError("trimesh not installed. Run: pip install trimesh")

    try:
        mesh = trimesh.load(str(usdz_path), force='mesh')

        if isinstance(mesh, trimesh.Scene):
            # Combine all meshes in scene
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if meshes:
                mesh = trimesh.util.concatenate(meshes)
            else:
                raise CollisionError("No mesh geometry found in USDZ")

        return {
            'vertices': np.array(mesh.vertices),
            'faces': np.array(mesh.faces),
            'bounds': {
                'min': mesh.bounds[0].tolist(),
                'max': mesh.bounds[1].tolist(),
            }
        }
    except Exception as e:
        raise CollisionError(f"Failed to parse USDZ with trimesh: {e}")


def simplify_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_ratio: float = 0.5,
    target_faces: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplify mesh to reduce polygon count.

    Args:
        vertices: Nx3 array of vertices
        faces: Mx3 array of triangle faces
        target_ratio: Target ratio of original face count
        target_faces: Explicit target face count (overrides ratio)

    Returns:
        Tuple of (simplified_vertices, simplified_faces)
    """
    try:
        import trimesh
        from trimesh.simplify import simplify_quadric_decimation
    except ImportError:
        console.print("[yellow]trimesh not available, skipping simplification[/yellow]")
        return vertices, faces

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    if target_faces is None:
        target_faces = int(len(faces) * target_ratio)

    # Use quadric decimation
    simplified = mesh.simplify_quadric_decimation(target_faces)

    console.print(f"[blue]Simplified mesh: {len(faces)} → {len(simplified.faces)} faces[/blue]")

    return np.array(simplified.vertices), np.array(simplified.faces)


def export_collision_glb(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: Path,
    include_normals: bool = True
) -> Path:
    """
    Export collision mesh to GLB format.

    Args:
        vertices: Nx3 array of vertices
        faces: Mx3 array of triangle faces
        output_path: Output path for GLB file
        include_normals: Whether to include vertex normals

    Returns:
        Path to exported GLB file
    """
    try:
        import trimesh
    except ImportError:
        raise CollisionError("trimesh not installed. Run: pip install trimesh")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    if include_normals:
        mesh.fix_normals()

    output_path = output_path.with_suffix('.glb')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh.export(str(output_path), file_type='glb')

    console.print(f"[green]Exported collision mesh to {output_path}[/green]")
    return output_path


def generate_floorplan_svg(
    floorplan_data: Dict,
    output_path: Path,
    width: int = 800,
    height: int = 600,
    padding: int = 50
) -> Path:
    """
    Generate SVG floorplan from wall/door/window data.

    Args:
        floorplan_data: Dict with walls, doors, windows lists
        output_path: Output path for SVG file
        width: SVG width in pixels
        height: SVG height in pixels
        padding: Padding around content

    Returns:
        Path to exported SVG file
    """
    walls = floorplan_data.get('walls', [])
    doors = floorplan_data.get('doors', [])
    windows = floorplan_data.get('windows', [])

    if not walls:
        console.print("[yellow]No wall data available for floorplan[/yellow]")
        # Create empty SVG
        svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>'
        output_path = output_path.with_suffix('.svg')
        output_path.write_text(svg_content)
        return output_path

    # Find bounds
    all_points = []
    for wall in walls:
        all_points.append(wall['start'])
        all_points.append(wall['end'])

    all_points = np.array(all_points)
    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)

    # Calculate scale to fit in SVG
    data_width = max_x - min_x
    data_height = max_y - min_y
    available_width = width - 2 * padding
    available_height = height - 2 * padding

    scale = min(available_width / data_width, available_height / data_height) if data_width > 0 and data_height > 0 else 1

    def transform(x, y):
        """Transform world coordinates to SVG coordinates."""
        sx = padding + (x - min_x) * scale
        sy = padding + (max_y - y) * scale  # Flip Y axis
        return sx, sy

    # Build SVG
    svg_elements = []

    # Walls
    for wall in walls:
        x1, y1 = transform(wall['start'][0], wall['start'][1])
        x2, y2 = transform(wall['end'][0], wall['end'][1])
        svg_elements.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#333" stroke-width="3" stroke-linecap="round"/>'
        )

    # Doors
    for door in doors:
        x, y = transform(door['position'][0], door['position'][1])
        w = door['width'] * scale
        svg_elements.append(
            f'<rect x="{x - w/2:.1f}" y="{y - 2:.1f}" width="{w:.1f}" height="4" '
            f'fill="#8B4513" stroke="none"/>'
        )

    # Windows
    for window in windows:
        x, y = transform(window['position'][0], window['position'][1])
        w = window['width'] * scale
        svg_elements.append(
            f'<rect x="{x - w/2:.1f}" y="{y - 2:.1f}" width="{w:.1f}" height="4" '
            f'fill="#87CEEB" stroke="#333" stroke-width="1"/>'
        )

    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f5f5f5"/>
  <g id="floorplan">
    {''.join(svg_elements)}
  </g>
</svg>'''

    output_path = output_path.with_suffix('.svg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg_content)

    console.print(f"[green]Generated floorplan SVG: {output_path}[/green]")
    return output_path


def generate_collision_from_usdz(
    usdz_path: Path,
    output_dir: Path,
    simplify: bool = True,
    target_faces: int = 10000
) -> Tuple[Path, Dict]:
    """
    Generate collision mesh from USDZ file.

    Args:
        usdz_path: Path to RoomPlan USDZ
        output_dir: Output directory
        simplify: Whether to simplify the mesh
        target_faces: Target face count after simplification

    Returns:
        Tuple of (collision_mesh_path, mesh_stats)
    """
    console.print(f"[blue]Processing USDZ: {usdz_path.name}[/blue]")

    # Parse USDZ
    mesh_data = parse_usdz_to_mesh(usdz_path)

    vertices = mesh_data['vertices']
    faces = mesh_data['faces']

    if len(vertices) == 0:
        raise CollisionError("No vertices found in USDZ")

    stats = {
        'original_vertices': len(vertices),
        'original_faces': len(faces) // 3 if len(faces) > 0 else 0,
    }

    # Reshape faces if needed (flat array to Nx3)
    if len(faces.shape) == 1 and len(faces) >= 3:
        faces = faces.reshape(-1, 3)

    # Simplify if requested
    if simplify and len(faces) > target_faces:
        vertices, faces = simplify_mesh(vertices, faces, target_faces=target_faces)

    stats['final_vertices'] = len(vertices)
    stats['final_faces'] = len(faces)

    # Export to GLB
    collision_path = export_collision_glb(
        vertices, faces,
        output_dir / "collision.glb"
    )

    stats['file_size'] = collision_path.stat().st_size

    return collision_path, stats


def generate_collision_from_ply(
    ply_path: Path,
    output_dir: Path,
    simplify: bool = True,
    target_faces: int = 10000
) -> Tuple[Path, Dict]:
    """
    Generate collision mesh from a PLY file (e.g., LiDAR mesh).

    Args:
        ply_path: Path to input PLY mesh file
        output_dir: Output directory
        simplify: Whether to simplify the mesh
        target_faces: Target face count after simplification

    Returns:
        Tuple of (collision_mesh_path, mesh_stats)
    """
    try:
        import trimesh
    except ImportError:
        raise CollisionError("trimesh not installed. Run: pip install trimesh")

    if not ply_path.exists():
        raise CollisionError(f"PLY file not found: {ply_path}")

    console.print(f"[blue]Processing PLY mesh: {ply_path.name}[/blue]")

    try:
        mesh = trimesh.load(str(ply_path), force='mesh')

        if isinstance(mesh, trimesh.Scene):
            # Combine all meshes in scene
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if meshes:
                mesh = trimesh.util.concatenate(meshes)
            else:
                raise CollisionError("No mesh geometry found in PLY")

        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

    except Exception as e:
        raise CollisionError(f"Failed to load PLY: {e}")

    stats = {
        'source': 'ply',
        'original_vertices': len(vertices),
        'original_faces': len(faces),
    }

    # Simplify if requested
    if simplify and len(faces) > target_faces:
        vertices, faces = simplify_mesh(vertices, faces, target_faces=target_faces)

    stats['final_vertices'] = len(vertices)
    stats['final_faces'] = len(faces)

    # Export to GLB
    collision_path = export_collision_glb(
        vertices, faces,
        output_dir / "collision.glb"
    )

    stats['file_size'] = collision_path.stat().st_size

    return collision_path, stats


def generate_collision_from_floorplan(
    floorplan_data: Dict,
    output_dir: Path,
    wall_thickness: float = 0.15,
    ceiling_height: float = 2.5
) -> Tuple[Path, Dict]:
    """
    Generate collision mesh from floorplan data (walls, doors, windows).

    This creates a simple box-based collision mesh from 2D floorplan.

    Args:
        floorplan_data: Dict with walls list
        output_dir: Output directory
        wall_thickness: Thickness of walls in meters
        ceiling_height: Height of walls in meters

    Returns:
        Tuple of (collision_mesh_path, mesh_stats)
    """
    walls = floorplan_data.get('walls', [])

    if not walls:
        raise CollisionError("No wall data in floorplan")

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for wall in walls:
        start = np.array(wall['start'])
        end = np.array(wall['end'])
        height = wall.get('height', ceiling_height)

        # Calculate wall direction and normal
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 0.01:
            continue

        direction = direction / length
        normal = np.array([-direction[1], direction[0]])  # Perpendicular in 2D

        # Create wall box vertices (8 corners)
        half_thickness = wall_thickness / 2

        # Bottom corners
        v0 = np.array([start[0] - normal[0] * half_thickness, 0, start[1] - normal[1] * half_thickness])
        v1 = np.array([start[0] + normal[0] * half_thickness, 0, start[1] + normal[1] * half_thickness])
        v2 = np.array([end[0] + normal[0] * half_thickness, 0, end[1] + normal[1] * half_thickness])
        v3 = np.array([end[0] - normal[0] * half_thickness, 0, end[1] - normal[1] * half_thickness])

        # Top corners
        v4 = v0 + np.array([0, height, 0])
        v5 = v1 + np.array([0, height, 0])
        v6 = v2 + np.array([0, height, 0])
        v7 = v3 + np.array([0, height, 0])

        vertices = [v0, v1, v2, v3, v4, v5, v6, v7]
        all_vertices.extend(vertices)

        # Create faces (2 triangles per face, 6 faces per box)
        faces = [
            # Front
            [0, 1, 5], [0, 5, 4],
            # Back
            [2, 3, 7], [2, 7, 6],
            # Left
            [0, 3, 7], [0, 7, 4],
            # Right
            [1, 2, 6], [1, 6, 5],
            # Top
            [4, 5, 6], [4, 6, 7],
            # Bottom
            [0, 2, 1], [0, 3, 2],
        ]

        # Offset face indices
        for face in faces:
            all_faces.append([f + vertex_offset for f in face])

        vertex_offset += 8

    vertices = np.array(all_vertices)
    faces = np.array(all_faces)

    stats = {
        'wall_count': len(walls),
        'vertices': len(vertices),
        'faces': len(faces),
    }

    # Export to GLB
    collision_path = export_collision_glb(
        vertices, faces,
        output_dir / "collision.glb"
    )

    stats['file_size'] = collision_path.stat().st_size

    return collision_path, stats


# CLI entry point
if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def main(
        work_dir: Path = typer.Argument(..., help="Work directory (from ingest stage)"),
        no_simplify: bool = typer.Option(False, help="Disable mesh simplification"),
        target_faces: int = typer.Option(10000, help="Target face count"),
    ):
        """Generate collision mesh and floorplan from work directory.

        Tries sources in order:
        1. USDZ room geometry (if available in manifest)
        2. PLY mesh (mesh.ply in package)
        3. Floorplan data (walls from manifest)
        """
        # Find manifest
        manifest_candidates = list(work_dir.rglob("scan_manifest.json"))
        if not manifest_candidates:
            console.print(f"[bold red]Error:[/bold red] No scan_manifest.json found in {work_dir}")
            raise typer.Exit(1)

        manifest_path = manifest_candidates[0]
        package_dir = manifest_path.parent

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        collision_path = None
        floorplan_path = None

        try:
            # Try USDZ first
            room_geometry = manifest.get('assets', {}).get('room_geometry')
            if room_geometry:
                usdz_path = package_dir / room_geometry
                if usdz_path.exists():
                    console.print("[blue]Using USDZ room geometry[/blue]")
                    collision_path, stats = generate_collision_from_usdz(
                        usdz_path, work_dir,
                        simplify=not no_simplify,
                        target_faces=target_faces
                    )

            # Try PLY mesh
            if collision_path is None:
                ply_candidates = list(package_dir.glob("*.ply")) + list(package_dir.glob("mesh.ply"))
                if ply_candidates:
                    ply_path = ply_candidates[0]
                    console.print(f"[blue]Using PLY mesh: {ply_path.name}[/blue]")
                    collision_path, stats = generate_collision_from_ply(
                        ply_path, work_dir,
                        simplify=not no_simplify,
                        target_faces=target_faces
                    )

            # Try floorplan data
            if collision_path is None:
                floorplan_data = manifest.get('floorplan', {})
                if floorplan_data.get('walls'):
                    console.print("[blue]Using floorplan wall data[/blue]")
                    collision_path, stats = generate_collision_from_floorplan(
                        floorplan_data, work_dir
                    )

            if collision_path:
                console.print(f"\n[bold green]Generated collision: {collision_path}[/bold green]")

            # Generate floorplan SVG if data available
            floorplan_data = manifest.get('floorplan', {})
            if floorplan_data.get('walls'):
                floorplan_path = generate_floorplan_svg(
                    floorplan_data,
                    work_dir / "floorplan.svg"
                )
                console.print(f"[green]Generated floorplan: {floorplan_path}[/green]")

            if not collision_path and not floorplan_path:
                console.print("[yellow]No collision or floorplan data available[/yellow]")

        except CollisionError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1)

    @app.command("from-usdz")
    def from_usdz(
        usdz_path: Path = typer.Argument(..., help="Path to RoomPlan USDZ"),
        output_dir: Path = typer.Option(Path("./output"), help="Output directory"),
        no_simplify: bool = typer.Option(False, help="Disable mesh simplification"),
        target_faces: int = typer.Option(10000, help="Target face count"),
    ):
        """Generate collision mesh from USDZ."""
        try:
            collision_path, stats = generate_collision_from_usdz(
                usdz_path,
                output_dir,
                simplify=not no_simplify,
                target_faces=target_faces
            )
            console.print(f"\n[bold green]Generated: {collision_path}[/bold green]")
            console.print(f"  Faces: {stats['final_faces']}")
        except CollisionError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1)

    @app.command("from-ply")
    def from_ply(
        ply_path: Path = typer.Argument(..., help="Path to PLY mesh"),
        output_dir: Path = typer.Option(Path("./output"), help="Output directory"),
        no_simplify: bool = typer.Option(False, help="Disable mesh simplification"),
        target_faces: int = typer.Option(10000, help="Target face count"),
    ):
        """Generate collision mesh from PLY."""
        try:
            collision_path, stats = generate_collision_from_ply(
                ply_path,
                output_dir,
                simplify=not no_simplify,
                target_faces=target_faces
            )
            console.print(f"\n[bold green]Generated: {collision_path}[/bold green]")
            console.print(f"  Faces: {stats['final_faces']}")
        except CollisionError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1)

    @app.command("floorplan")
    def floorplan(
        manifest_path: Path = typer.Argument(..., help="Path to scan_manifest.json"),
        output_dir: Path = typer.Option(Path("./output"), help="Output directory"),
    ):
        """Generate floorplan SVG from manifest."""
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        floorplan_data = manifest.get('floorplan', {})
        if not floorplan_data:
            console.print("[yellow]No floorplan data in manifest[/yellow]")
            raise typer.Exit(1)

        svg_path = generate_floorplan_svg(
            floorplan_data,
            output_dir / "floorplan.svg"
        )
        console.print(f"\n[bold green]Generated: {svg_path}[/bold green]")

    app()
