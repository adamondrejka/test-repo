#!/bin/bash
# Quick file upload to RunPod via Cloudflare Tunnel
# Usage: ./upload_to_pod.sh <file_or_directory>
#
# Requires: brew install cloudflared

set -e

FILE=${1:?"Usage: $0 <file_or_directory>"}
PORT=${2:-8765}

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "❌ cloudflared not found. Install with:"
    echo "   brew install cloudflared"
    exit 1
fi

# Get absolute path and filename
if [[ "$FILE" = /* ]]; then
    FILEPATH="$FILE"
else
    FILEPATH="$(pwd)/$FILE"
fi
FILENAME=$(basename "$FILEPATH")
FILEDIR=$(dirname "$FILEPATH")

if [ ! -e "$FILEPATH" ]; then
    echo "❌ File not found: $FILEPATH"
    exit 1
fi

# Show file info
if [ -f "$FILEPATH" ]; then
    FILESIZE=$(ls -lh "$FILEPATH" | awk '{print $5}')
    echo "📦 File: $FILENAME ($FILESIZE)"
else
    echo "📁 Directory: $FILENAME"
fi

echo ""
echo "🚀 Starting upload server..."
echo ""

# Start HTTP server in background
cd "$FILEDIR"
python3 -m http.server $PORT &
HTTP_PID=$!

# Cleanup on exit
cleanup() {
    kill $HTTP_PID 2>/dev/null || true
    echo ""
    echo "🛑 Server stopped"
}
trap cleanup EXIT

# Give server time to start
sleep 1

# Run cloudflared and capture URL
echo "🌐 Creating tunnel (this may take a few seconds)..."
echo ""

# Run cloudflared and parse the URL
cloudflared tunnel --url http://localhost:$PORT 2>&1 | while read line; do
    # Look for the tunnel URL
    if [[ "$line" == *"trycloudflare.com"* ]]; then
        URL=$(echo "$line" | grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com')
        if [ -n "$URL" ]; then
            echo "════════════════════════════════════════════════════════════"
            echo ""
            echo "✅ Tunnel ready!"
            echo ""
            echo "📋 Run this on RunPod:"
            echo ""
            echo "   wget ${URL}/${FILENAME} -O /workspace/${FILENAME}"
            echo ""
            echo "════════════════════════════════════════════════════════════"
            echo ""
            echo "Press Ctrl+C when download is complete"
            echo ""
        fi
    fi
done
