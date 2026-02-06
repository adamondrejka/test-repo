"""
Reality Engine Backend Pipeline

Converts iOS scan packages (video + ARKit poses + RoomPlan geometry)
into web-ready Gaussian Splat tour packages.

Pipeline stages:
1. Ingest - Unzip and validate scan package
2. Extract Frames - Video → images with pose matching
3. Convert Poses - ARKit → Nerfstudio format
4. Train - Gaussian Splat training
5. Compress - PLY → SPZ compression
6. Collision - Generate collision mesh from RoomPlan
7. Package - Bundle final tour package
"""

__version__ = "0.1.0"
