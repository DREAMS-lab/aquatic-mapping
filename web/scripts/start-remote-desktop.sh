#!/bin/bash
# Start Remote Desktop Access for Field Sampler
# Supports dual monitors via x11vnc + noVNC

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$LOG_DIR"

# Configuration
VNC_PORT=5900
NOVNC_PORT=6080
DISPLAY_NUM=":0"

# Check if already running
if pgrep -f "x11vnc.*-rfbport $VNC_PORT" > /dev/null; then
    echo "x11vnc already running on port $VNC_PORT"
else
    echo "Starting x11vnc on port $VNC_PORT..."

    # Start x11vnc with dual monitor support
    # -clip xinerama0 for first monitor, xinerama1 for second, or remove for both
    x11vnc \
        -display $DISPLAY_NUM \
        -forever \
        -shared \
        -rfbport $VNC_PORT \
        -rfbauth /home/blazair/.vnc/passwd \
        -xkb \
        -noxrecord \
        -noxfixes \
        -noxdamage \
        -repeat \
        -bg \
        -o "$LOG_DIR/x11vnc.log"

    echo "x11vnc started (dual monitor support enabled)"
fi

# Check if noVNC already running
if pgrep -f "websockify.*$NOVNC_PORT" > /dev/null; then
    echo "noVNC already running on port $NOVNC_PORT"
else
    echo "Starting noVNC web server on port $NOVNC_PORT..."

    # Start noVNC websockify proxy
    websockify --web=/usr/share/novnc/ $NOVNC_PORT localhost:$VNC_PORT > "$LOG_DIR/novnc.log" 2>&1 &

    echo "noVNC started"
fi

echo ""
echo "=============================================="
echo "  Remote Desktop Access Ready!"
echo "=============================================="
echo ""
echo "  Access from browser:"
echo "    http://localhost:$NOVNC_PORT/vnc.html"
echo ""
echo "  Or connect with VNC client:"
echo "    localhost:$VNC_PORT"
echo ""
echo "  For remote access via Field Sampler:"
echo "    Use the Host PC tab in the dashboard"
echo ""
echo "=============================================="
