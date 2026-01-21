#!/bin/bash
# Stop Remote Desktop Access

echo "Stopping remote desktop services..."

# Stop x11vnc
pkill -f "x11vnc.*-rfbport 5900" 2>/dev/null && echo "x11vnc stopped" || echo "x11vnc was not running"

# Stop noVNC
pkill -f "websockify.*6080" 2>/dev/null && echo "noVNC stopped" || echo "noVNC was not running"

echo "Remote desktop services stopped."
