#!/bin/bash
# Cloud Training Script for Reality Engine
# Usage: ./cloud_train.sh <scan.zip> [iterations]

set -e

SCAN_FILE=${1:?"Usage: $0 <scan.zip> [iterations]"}
ITERATIONS=${2:-30000}
WORKSPACE=/workspace
OUTPUT_DIR=$WORKSPACE/output

echo "=========================================="
echo "Reality Engine - Cloud Training"
echo "=========================================="
echo "Scan: $SCAN_FILE"
echo "Iterations: $ITERATIONS"
echo ""

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Install dependencies if needed
if ! command -v ns-train &> /dev/null; then
    echo "Installing nerfstudio..."
    pip install nerfstudio gsplat
fi

# Check ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg..."
    apt-get update && apt-get install -y ffmpeg
fi

# Install backend if needed
if ! python -c "import pipeline" 2>/dev/null; then
    echo "Installing Reality Engine backend..."
    cd $WORKSPACE/backend 2>/dev/null || cd /workspace/reality/backend
    pip install -e .
fi

# Run pipeline
echo ""
echo "Starting processing pipeline..."
echo ""

python -m pipeline.process "$SCAN_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --iterations "$ITERATIONS"

# Find output
TOUR_DIR=$(ls -td $OUTPUT_DIR/tour_* 2>/dev/null | head -1)

if [ -n "$TOUR_DIR" ]; then
    echo ""
    echo "=========================================="
    echo "Training complete!"
    echo "=========================================="
    echo "Output: $TOUR_DIR"
    echo ""
    echo "Contents:"
    ls -lh "$TOUR_DIR"
    echo ""

    # Create downloadable zip
    ZIP_NAME="tour_package_$(date +%Y%m%d_%H%M%S).zip"
    cd $OUTPUT_DIR
    zip -r "$ZIP_NAME" "$(basename $TOUR_DIR)"
    echo "Download: $OUTPUT_DIR/$ZIP_NAME"
    echo ""
    echo "To download, run on your local machine:"
    echo "  scp -P <PORT> root@<IP>:$OUTPUT_DIR/$ZIP_NAME ."
else
    echo "Error: No output found"
    exit 1
fi
