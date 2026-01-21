#!/bin/bash
# =============================================================================
# Field Sampler - Stop Everything
# =============================================================================

echo ""
echo "=============================================="
echo "  Field Sampler - Stopping All Services"
echo "=============================================="
echo ""

echo "[1/4] Stopping web server..."
pkill -f "uvicorn backend.main:app" 2>/dev/null && echo "  ✓ Stopped" || echo "  - Not running"

echo "[2/4] Stopping host VNC..."
pkill -f "x11vnc.*-rfbport 5900" 2>/dev/null && echo "  ✓ x11vnc stopped" || echo "  - x11vnc not running"
pkill -f "websockify.*6080" 2>/dev/null && echo "  ✓ noVNC stopped" || echo "  - noVNC not running"

echo "[3/4] Stopping trial VNC proxy..."
pkill -f "trial-vnc-proxy.py" 2>/dev/null && echo "  ✓ Stopped" || echo "  - Not running"

echo "[4/4] Stopping Cloudflare tunnel..."
sudo systemctl stop cloudflared 2>/dev/null && echo "  ✓ Stopped" || pkill -f "cloudflared.*tunnel" 2>/dev/null && echo "  ✓ Stopped" || echo "  - Not running"

echo ""
echo "=============================================="
echo "  All Services Stopped"
echo "=============================================="
echo ""
