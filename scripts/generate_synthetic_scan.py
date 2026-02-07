#!/usr/bin/env python3
"""
Generate a synthetic iOS scan package of a colored cube.

Creates a fake scan_manifest.json + rendered JPEG images with known-correct
ARKit-convention camera poses. Use this to verify the backend pipeline
independently of real iOS data.

Usage:
    python scripts/generate_synthetic_scan.py [output_dir]

Then run the pipeline:
    python -m pipeline.process synthetic_scan/ output/synthetic_test/
"""

import json
import math
import numpy as np
import zipfile
from PIL import Image, ImageDraw
from pathlib import Path
import sys


# ── Scene: colored cube ──────────────────────────────────────────────

CUBE_HALF = 0.5  # half-size in meters (1m cube)

VERTICES = np.array([
    [-1, -1, -1],  # 0
    [ 1, -1, -1],  # 1
    [ 1,  1, -1],  # 2
    [-1,  1, -1],  # 3
    [-1, -1,  1],  # 4
    [ 1, -1,  1],  # 5
    [ 1,  1,  1],  # 6
    [-1,  1,  1],  # 7
], dtype=float) * CUBE_HALF

# (vertex_indices, RGB color, face name)
FACES = [
    ([4, 5, 6, 7], (220,  40,  40), "+Z front  RED"),
    ([1, 0, 3, 2], ( 40, 200,  40), "-Z back   GREEN"),
    ([5, 1, 2, 6], ( 40,  40, 220), "+X right  BLUE"),
    ([0, 4, 7, 3], (220, 220,  40), "-X left   YELLOW"),
    ([7, 6, 2, 3], ( 40, 220, 220), "+Y top    CYAN"),
    ([0, 1, 5, 4], (220,  40, 220), "-Y bottom MAGENTA"),
]

# Checkerboard ground plane at y = -CUBE_HALF
GROUND_Y = -CUBE_HALF
GROUND_HALF = 3.0  # extends ±3m
TILE_SIZE = 0.5
GROUND_COLOR_A = (180, 180, 180)
GROUND_COLOR_B = (120, 120, 120)


# ── Camera settings ──────────────────────────────────────────────────

IMAGE_W = 640
IMAGE_H = 480
FX = FY = 500.0
CX = IMAGE_W / 2.0
CY = IMAGE_H / 2.0

NUM_RINGS = 3
CAMERAS_PER_RING = 12
CAMERA_RADIUS = 3.0
RING_HEIGHTS = [0.0, 0.8, -0.4]  # relative to cube center
TARGET = np.array([0.0, 0.0, 0.0])  # look at cube center


# ── Math helpers ─────────────────────────────────────────────────────

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def look_at_arkit(eye, target, world_up=np.array([0.0, 1.0, 0.0])):
    """
    Build a camera-to-world matrix in ARKit convention.

    ARKit camera space: X-right, Y-up, Z-backward (looks along -Z).
    """
    forward = normalize(target - eye)       # direction camera looks
    right   = normalize(np.cross(forward, world_up))
    up      = np.cross(right, forward)      # guaranteed unit if above are unit

    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward   # Z = backward
    c2w[:3, 3] = eye
    return c2w


def project_arkit(p_world, w2c):
    """
    Project a world point to pixel coords using ARKit camera convention.

    Returns (u, v, z_cam) or None if behind camera.
    """
    p = w2c[:3, :3] @ p_world + w2c[:3, 3]
    if p[2] >= -1e-6:
        return None  # behind camera (ARKit: visible = z < 0)
    u = FX * p[0] / (-p[2]) + CX
    v = FY * (-p[1]) / (-p[2]) + CY   # flip Y: camera-up → image-down
    return (u, v, p[2])


# ── Rendering ────────────────────────────────────────────────────────

def face_normal(vertex_indices):
    """Outward-facing normal of a quad face."""
    v0, v1, v2 = VERTICES[vertex_indices[0]], VERTICES[vertex_indices[1]], VERTICES[vertex_indices[2]]
    n = np.cross(v1 - v0, v2 - v0)
    return normalize(n)


def render_ground(draw, w2c, cam_pos):
    """Draw a checkerboard ground plane using the painter's algorithm."""
    tiles = []
    steps = int(GROUND_HALF / TILE_SIZE)
    for ix in range(-steps, steps):
        for iz in range(-steps, steps):
            x0 = ix * TILE_SIZE
            z0 = iz * TILE_SIZE
            corners_world = [
                np.array([x0,             GROUND_Y, z0]),
                np.array([x0 + TILE_SIZE, GROUND_Y, z0]),
                np.array([x0 + TILE_SIZE, GROUND_Y, z0 + TILE_SIZE]),
                np.array([x0,             GROUND_Y, z0 + TILE_SIZE]),
            ]

            projected = []
            for cw in corners_world:
                p = project_arkit(cw, w2c)
                if p is None:
                    break
                projected.append(p)
            if len(projected) != 4:
                continue

            avg_z = np.mean([p[2] for p in projected])
            color = GROUND_COLOR_A if (ix + iz) % 2 == 0 else GROUND_COLOR_B
            pixels = [(int(p[0]), int(p[1])) for p in projected]
            tiles.append((avg_z, pixels, color))

    # Painter's order (furthest first = most negative z first)
    tiles.sort(key=lambda t: t[0])
    for _, pixels, color in tiles:
        draw.polygon(pixels, fill=color)


def render_frame(c2w):
    """Render the cube + ground from the given ARKit C2W pose."""
    w2c = np.linalg.inv(c2w)
    cam_pos = c2w[:3, 3]

    img = Image.new("RGB", (IMAGE_W, IMAGE_H), (210, 220, 230))  # sky
    draw = ImageDraw.Draw(img)

    # ── collect all drawable faces (cube + ground) with depth ──
    drawables = []  # (avg_z, pixels, color)

    # Ground tiles
    render_ground(draw, w2c, cam_pos)

    # Cube faces
    for vi, color, _name in FACES:
        # Back-face culling
        n = face_normal(vi)
        face_center = VERTICES[vi].mean(axis=0)
        if np.dot(n, cam_pos - face_center) <= 0:
            continue

        projected = []
        for idx in vi:
            p = project_arkit(VERTICES[idx], w2c)
            if p is None:
                break
            projected.append(p)
        if len(projected) != 4:
            continue

        avg_z = np.mean([p[2] for p in projected])
        pixels = [(int(p[0]), int(p[1])) for p in projected]
        drawables.append((avg_z, pixels, color))

    # Draw cube faces on top (painter's: furthest first)
    drawables.sort(key=lambda d: d[0])
    for _, pixels, color in drawables:
        draw.polygon(pixels, fill=color)

    return img


# ── Camera generation ────────────────────────────────────────────────

def generate_cameras():
    """Place cameras on rings around the cube, looking inward."""
    cameras = []
    for height in RING_HEIGHTS:
        for i in range(CAMERAS_PER_RING):
            angle = 2 * math.pi * i / CAMERAS_PER_RING
            x = CAMERA_RADIUS * math.cos(angle)
            z = CAMERA_RADIUS * math.sin(angle)
            pos = np.array([x, height, z])
            cameras.append(look_at_arkit(pos, TARGET))
    return cameras


# ── Manifest generation ──────────────────────────────────────────────

def c2w_to_row_major_16(c2w):
    """Flatten 4×4 → 16 floats in row-major order (matching iOS toRowMajorArray)."""
    return [float(x) for x in c2w.flatten()]


def build_manifest(poses_data):
    """Build a scan_manifest.json dict matching the iOS ScanManifest format."""
    # 3×3 intrinsic matrix as 9-element row-major list
    #   [fx,  0, cx,
    #     0, fy, cy,
    #     0,  0,  1]
    intrinsic_matrix = [
        FX, 0.0, CX,
        0.0, FY, CY,
        0.0, 0.0, 1.0,
    ]

    return {
        "scan_id": "synthetic-cube-test",
        "timestamp": "2025-01-01T00:00:00Z",
        "scan_type": "INDOOR_HYBRID",
        "calibration": {
            "intrinsic_matrix": [float(v) for v in intrinsic_matrix],
            "image_width": IMAGE_W,
            "image_height": IMAGE_H,
        },
        "assets": {
            "images_dir": "images",
        },
        "poses": poses_data,
        "capture_orientation": "landscape",
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("synthetic_scan")
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    cameras = generate_cameras()
    print(f"Generating {len(cameras)} synthetic views of a colored cube …")

    poses_data = []
    for i, c2w in enumerate(cameras):
        fname = f"frame_{i:06d}.jpg"

        # Render & save
        img = render_frame(c2w)
        img.save(images_dir / fname, quality=92)

        # Verify rotation matrix
        R = c2w[:3, :3]
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-6, f"Frame {i}: det(R)={det}"

        poses_data.append({
            "timestamp": float(i) * 0.2,
            "transform_matrix": c2w_to_row_major_16(c2w),
            "tracking_state": "normal",
            "image_name": fname,
        })

    # Write manifest
    manifest = build_manifest(poses_data)
    with open(out / "scan_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Zip it (like the iOS app would) ──
    zip_path = out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out / "scan_manifest.json", "scan_manifest.json")
        for img_file in sorted(images_dir.glob("frame_*.jpg")):
            zf.write(img_file, f"images/{img_file.name}")

    # ── Summary ──
    zip_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\nSynthetic scan package: {zip_path}  ({zip_mb:.1f} MB)")
    print(f"  {len(cameras)} frames  |  {IMAGE_W}×{IMAGE_H}  |  fx=fy={FX}")
    print(f"  Cube: 1 m, faces: RED(+Z) GREEN(-Z) BLUE(+X) YELLOW(-X) CYAN(+Y) MAGENTA(-Y)")
    print(f"  Cameras: radius={CAMERA_RADIUS} m, {NUM_RINGS} rings × {CAMERAS_PER_RING}")
    print(f"  Ground: checkerboard at y={GROUND_Y}")
    print(f"\nPoses are in ARKit convention (Y-up, camera looks -Z, C2W).")
    print(f"Run pipeline:  python -m pipeline.process {zip_path} output/synthetic_test/")


if __name__ == "__main__":
    main()
