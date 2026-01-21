#!/bin/bash
# =============================================================================
# Field Sampler - Start Everything
# Run this script after booting your PC to start all services
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "=============================================="
echo "  Field Sampler - Starting All Services"
echo "=============================================="
echo ""

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# -----------------------------------------------------------------------------
# 1. Main Web Server (port 8000)
# -----------------------------------------------------------------------------
echo "[1/4] Starting web server..."
if pgrep -f "uvicorn backend.main:app" > /dev/null; then
    echo "  ✓ Already running"
else
    cd "$SCRIPT_DIR"
    nohup python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 > "$SCRIPT_DIR/logs/server.log" 2>&1 &
    sleep 2
    echo "  ✓ Started on port 8000"
fi

# -----------------------------------------------------------------------------
# 2. Host VNC (x11vnc + noVNC on port 6080)
# -----------------------------------------------------------------------------
echo "[2/4] Starting host remote desktop..."

# Start x11vnc if not running
if pgrep -f "x11vnc.*-rfbport 5900" > /dev/null; then
    echo "  ✓ x11vnc already running"
else
    if [ -f ~/.vnc/passwd ]; then
        x11vnc -display :0 -forever -shared -rfbport 5900 -rfbauth ~/.vnc/passwd -bg -o "$SCRIPT_DIR/logs/x11vnc.log" 2>/dev/null || true
    else
        x11vnc -display :0 -forever -shared -rfbport 5900 -nopw -bg -o "$SCRIPT_DIR/logs/x11vnc.log" 2>/dev/null || true
    fi
    echo "  ✓ x11vnc started"
fi

# Start noVNC websockify if not running
if pgrep -f "websockify.*6080" > /dev/null; then
    echo "  ✓ noVNC already running"
else
    websockify --web=/usr/share/novnc/ 6080 localhost:5900 > "$SCRIPT_DIR/logs/novnc.log" 2>&1 &
    echo "  ✓ noVNC started on port 6080"
fi

# -----------------------------------------------------------------------------
# 3. Trial VNC Proxy (port 6099 - routes trialX.domain.com to port 608X)
# -----------------------------------------------------------------------------
echo "[3/4] Starting trial VNC proxy..."
if pgrep -f "trial-vnc-proxy.py" > /dev/null; then
    echo "  ✓ Already running"
else
    cd "$SCRIPT_DIR"
    nohup python scripts/trial-vnc-proxy.py > "$SCRIPT_DIR/logs/trial-proxy.log" 2>&1 &
    sleep 1
    echo "  ✓ Started on port 6099"
fi

# -----------------------------------------------------------------------------
# 4. Cloudflare Tunnel
# -----------------------------------------------------------------------------
echo "[4/4] Starting Cloudflare tunnel..."
if pgrep -f "cloudflared.*tunnel" > /dev/null; then
    echo "  ✓ Already running"
else
    # Try systemd first, fall back to manual
    if sudo systemctl start cloudflared 2>/dev/null; then
        echo "  ✓ Started via systemd"
    else
        echo "  Starting manually (keep this terminal open or run in tmux)..."
        cloudflared tunnel run bharath-site &
        echo "  ✓ Started manually"
    fi
fi

# -----------------------------------------------------------------------------
# Done!
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  All Services Started!"
echo "=============================================="
echo ""
echo "  LOCAL ACCESS:"
echo "    Landing Page:    http://localhost:8000"
echo "    Dashboard:       http://localhost:8000/dashboard"
echo "    Host VNC:        http://localhost:6080/vnc.html"
echo ""
echo "  REMOTE ACCESS (via Cloudflare):"
echo "    Landing Page:    https://bharathdesikan.com"
echo "    Dashboard:       https://bharathdesikan.com/dashboard"
echo "    Host VNC:        https://vnc.bharathdesikan.com/vnc.html"
echo "    Trial VNC:       https://trial4.bharathdesikan.com/vnc.html"
echo ""
echo "  LOGS:"
echo "    tail -f logs/server.log"
echo "    tail -f logs/trial-proxy.log"
echo ""
echo "=============================================="
