"""
Depth Supervision from LiDAR Mesh

Renders depth maps from a LiDAR mesh (mesh.ply) at each camera pose.
These depth maps guide Gaussian Splat training for better geometry.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from rich.console import Console

console = Console()


class DepthError(Exception):
    """Error during depth map generation."""
    pass


def generate_depth_maps(
    mesh_path: Path,
    transforms_path: Path,
    output_dir: Path,
    image_width: int,
    image_height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> int:
    """
    Render depth maps from a mesh at each camera pose in transforms.json.

    Depth maps are saved as 16-bit PNG files (millimeters) which nerfstudio
    can read with depth_unit_scale_factor=0.001.

    Args:
        mesh_path: Path to LiDAR mesh (PLY format)
        transforms_path: Path to transforms.json
        output_dir: Directory to save depth maps
        image_width, image_height: Image dimensions
        fx, fy, cx, cy: Camera intrinsics

    Returns:
        Number of depth maps generated
    """
    import json
    import trimesh

    console.print(f"[blue]Loading mesh from {mesh_path.name}...[/blue]")
    mesh = trimesh.load(str(mesh_path), force='mesh')

    if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
        raise DepthError(f"Mesh has no faces: {mesh_path}")

    console.print(f"[dim]Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces[/dim]")

    with open(transforms_path) as f:
        transforms = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    # Build ray caster from mesh
    ray_caster = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    for i, frame in enumerate(transforms['frames']):
        c2w = np.array(frame['transform_matrix'])
        file_path = frame.get('file_path', '')
        frame_name = Path(file_path).stem

        depth_map = _render_depth_at_pose(
            ray_caster, c2w,
            image_width, image_height,
            fx, fy, cx, cy,
            downsample=4,  # Render at 1/4 res for speed, nerfstudio upsamples
        )

        if depth_map is not None:
            depth_name = frame_name.replace('frame_', 'depth_') + '.png'
            depth_path = output_dir / depth_name
            _save_depth_png(depth_map, depth_path)
            count += 1

        if (i + 1) % 50 == 0:
            console.print(f"[dim]  Generated {i + 1}/{len(transforms['frames'])} depth maps[/dim]")

    console.print(f"[green]Generated {count} depth maps in {output_dir}[/green]")
    return count


def _render_depth_at_pose(
    ray_caster,
    c2w: np.ndarray,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    downsample: int = 4,
) -> Optional[np.ndarray]:
    """Render a depth map by casting rays through each pixel."""
    ds_w = width // downsample
    ds_h = height // downsample
    ds_fx = fx / downsample
    ds_fy = fy / downsample
    ds_cx = cx / downsample
    ds_cy = cy / downsample

    # Generate ray directions in camera space (OpenGL: -Z forward)
    u = np.arange(ds_w)
    v = np.arange(ds_h)
    uu, vv = np.meshgrid(u, v)
    dirs_cam = np.stack([
        (uu - ds_cx) / ds_fx,
        (vv - ds_cy) / ds_fy,
        -np.ones_like(uu),  # -Z forward in OpenGL
    ], axis=-1)

    # Reshape to (N, 3)
    dirs_cam = dirs_cam.reshape(-1, 3)
    # Normalize
    dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=1, keepdims=True)

    # Transform to world space
    rotation = c2w[:3, :3]
    origin = c2w[:3, 3]
    dirs_world = (rotation @ dirs_cam.T).T

    origins = np.tile(origin, (len(dirs_world), 1))

    try:
        locations, index_ray, index_tri = ray_caster.intersects_location(
            origins, dirs_world, multiple_hits=False
        )
    except Exception:
        return None

    if len(locations) == 0:
        return None

    # Compute depths
    depth_map = np.zeros(ds_w * ds_h, dtype=np.float32)
    distances = np.linalg.norm(locations - origins[index_ray], axis=1)
    depth_map[index_ray] = distances

    return depth_map.reshape(ds_h, ds_w)


def _save_depth_png(depth: np.ndarray, path: Path) -> None:
    """Save depth map as 16-bit PNG (values in millimeters)."""
    import cv2
    # Convert meters to millimeters, clamp to uint16 range
    depth_mm = (depth * 1000.0).clip(0, 65535).astype(np.uint16)
    cv2.imwrite(str(path), depth_mm)


def check_depth_support(transforms_path: Path) -> bool:
    """Check if any frames have depth_file_path set."""
    import json
    with open(transforms_path) as f:
        transforms = json.load(f)
    return any(
        'depth_file_path' in frame
        for frame in transforms.get('frames', [])
    )
