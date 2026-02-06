# Reality Engine Backend

Python backend for processing iOS scan packages into web-ready Gaussian Splat tour packages.

## Features

- **Frame Extraction**: FFmpeg-based video frame extraction with pose matching
- **Pose Conversion**: ARKit to Nerfstudio transforms.json conversion
- **Gaussian Splat Training**: Integration with Nerfstudio/gsplat
- **Compression**: PLY to SPZ format (~90% size reduction)
- **Collision Generation**: USDZ to GLB mesh conversion
- **Floorplan SVG**: Generate 2D floorplans from RoomPlan data

## Requirements

- Python 3.10+
- FFmpeg
- CUDA GPU (recommended for training)
- Docker (optional, for containerized deployment)

## Installation

### Local Installation

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev]"

# For training capabilities (requires CUDA)
pip install -e ".[training]"
```

### Docker

```bash
# Build image
docker build -t reality-backend .

# Run
docker run --gpus all -v ./data:/data reality-backend
```

## Project Structure

```
backend/
├── pyproject.toml          # Project configuration
├── Dockerfile              # Container definition
├── pipeline/
│   ├── __init__.py
│   ├── process.py          # Main orchestrator CLI
│   ├── ingest.py           # ZIP extraction and validation
│   ├── extract_frames.py   # Video to frames
│   ├── convert_poses.py    # ARKit to Nerfstudio
│   ├── train.py            # Gaussian Splat training
│   ├── compress.py         # PLY to SPZ
│   ├── collision.py        # Collision mesh generation
│   └── package.py          # Tour package bundling
├── utils/
│   ├── __init__.py
│   ├── matrix.py           # Matrix transformation utilities
│   └── validation.py       # Input validation
└── tests/
    ├── test_poses.py       # Matrix conversion tests
    └── test_pipeline.py    # Integration tests
```

## Usage

### Full Pipeline

```bash
# Process a complete scan package
python -m pipeline.process /path/to/scan.zip --output-dir ./output

# With custom settings
python -m pipeline.process /path/to/scan.zip \
  --output-dir ./output \
  --iterations 50000 \
  --compression high
```

### Individual Stages

```bash
# Ingest and validate
python -m pipeline.ingest /path/to/scan.zip --work-dir ./work

# Extract frames
python -m pipeline.extract_frames video.mov poses.json --output-dir ./frames

# Convert poses
python -m pipeline.convert_poses scan_manifest.json ./frames -o transforms.json

# Train (requires GPU)
python -m pipeline.train ./train_data --output-dir ./model --iterations 30000

# Compress
python -m pipeline.compress model/splat.ply -o scene.spz

# Generate collision
python -m pipeline.collision from-usdz room.usdz --output-dir ./collision

# Package
python -m pipeline.package scan-id manifest.json scene.spz --output-dir ./package
```

### Pipeline Stages

```bash
# List all stages
python -m pipeline.process stages
```

Output:
```
1. Ingest      - Unzip and validate scan package
2. Extract     - Extract video frames and match to poses
3. Convert     - Convert ARKit poses to Nerfstudio format
4. Train       - Train Gaussian Splat model
5. Compress    - Compress PLY to SPZ format
6. Collision   - Generate collision mesh and floorplan
7. Package     - Bundle final tour package
```

## Output Format

The pipeline produces a tour package directory:

```
tour_[scan-id]/
├── metadata.json      # Tour metadata and viewer config
├── scene.spz          # Compressed Gaussian Splat
├── collision.glb      # Collision mesh (optional)
├── floorplan.svg      # 2D floorplan (optional)
└── thumbnail.jpg      # Preview image
```

### metadata.json Schema

```json
{
  "version": "1.0",
  "scan_id": "uuid",
  "created_at": "2024-01-15T10:30:00Z",
  "original_scan": {
    "timestamp": "...",
    "scan_type": "INDOOR_HYBRID",
    "calibration": { "width": 1920, "height": 1080 },
    "frame_count": 500
  },
  "assets": {
    "splat": "scene.spz",
    "collision": "collision.glb",
    "floorplan": "floorplan.svg",
    "thumbnail": "thumbnail.jpg"
  },
  "viewer": {
    "start_position": [0, 1.6, 0],
    "camera_height": 1.6,
    "movement_speed": 2.0,
    "collision_enabled": true
  },
  "bounds": {
    "min": [-5, 0, -5],
    "max": [5, 3, 5]
  }
}
```

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=pipeline --cov=utils

# Specific test file
pytest tests/test_poses.py -v
```

## Configuration

### Training Parameters

Edit `TrainingConfig` in `pipeline/train.py`:

```python
@dataclass
class TrainingConfig:
    max_iterations: int = 30000
    warmup_iterations: int = 500
    densify_until: int = 15000
    # ... more options
```

### Environment Variables

- `DATA_INPUT`: Input data directory (Docker)
- `DATA_OUTPUT`: Output data directory (Docker)
- `DATA_TEMP`: Temporary files directory (Docker)
- `CUDA_VISIBLE_DEVICES`: GPU selection

## Tech Stack

- **Python 3.10+**: Core language
- **Nerfstudio**: Gaussian Splat training
- **gsplat**: Fast splat rendering
- **FFmpeg**: Video processing
- **trimesh**: Mesh operations
- **Typer/Rich**: CLI interface
- **Pydantic**: Data validation
