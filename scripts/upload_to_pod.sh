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

# Start HTTP server in background
cd "$FILEDIR"
python3 -m http.server $PORT 2>/dev/null &
HTTP_PID=$!

# Cleanup on exit
cleanup() {
    kill $HTTP_PID 2>/dev/null || true
    kill $TUNNEL_PID 2>/dev/null || true
    echo ""
    echo "🛑 Server stopped"
}
trap cleanup EXIT INT TERM

# Give server time to start
sleep 1

echo "🌐 Creating tunnel..."
echo ""

# Create a temp file for cloudflared output
TMPFILE=$(mktemp)

# Run cloudflared in background, capture output
cloudflared tunnel --url http://localhost:$PORT > "$TMPFILE" 2>&1 &
TUNNEL_PID=$!

# Wait for URL to appear (max 30 seconds)
for i in {1..30}; do
    if grep -q "trycloudflare.com" "$TMPFILE" 2>/dev/null; then
        URL=$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' "$TMPFILE" | head -1)
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
            break
        fi
    fi
    sleep 1
done

# Check if we got the URL
if [ -z "$URL" ]; then
    echo "❌ Failed to get tunnel URL. Cloudflared output:"
    cat "$TMPFILE"
    rm "$TMPFILE"
    exit 1
fi

rm "$TMPFILE"

# Wait for user to press Ctrl+C
wait $TUNNEL_PID
