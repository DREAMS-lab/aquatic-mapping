#!/bin/bash
# =============================================================================
# Remote Desktop Setup Script for Field Sampler
# Enables web-based access to host PC with dual monitor support
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "  Field Sampler - Remote Desktop Setup"
echo "=============================================="
echo ""

# Check if running as root for package installation
if [ "$EUID" -eq 0 ]; then
    echo "Please don't run as root. Script will ask for sudo when needed."
    exit 1
fi

# Install required packages
echo "[1/4] Installing required packages..."
sudo apt-get update
sudo apt-get install -y x11vnc novnc websockify

# Create VNC password (optional, can use -nopw for no password)
echo ""
echo "[2/4] Setting up VNC..."
echo "Do you want to set a VNC password? (recommended for security)"
read -p "Set password? (y/n): " SET_PWD
if [ "$SET_PWD" = "y" ]; then
    mkdir -p ~/.vnc
    x11vnc -storepasswd ~/.vnc/passwd
    VNC_AUTH="-rfbauth $HOME/.vnc/passwd"
else
    VNC_AUTH="-nopw"
    echo "Warning: VNC will run without password (protected by web auth only)"
fi

# Create start script
echo ""
echo "[3/4] Creating start/stop scripts..."

cat > "$WEB_DIR/scripts/start-remote-desktop.sh" << 'STARTSCRIPT'
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
        VNC_AUTH_PLACEHOLDER \
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
STARTSCRIPT

# Replace auth placeholder
if [ -f ~/.vnc/passwd ]; then
    sed -i "s|VNC_AUTH_PLACEHOLDER|-rfbauth $HOME/.vnc/passwd|" "$WEB_DIR/scripts/start-remote-desktop.sh"
else
    sed -i "s|VNC_AUTH_PLACEHOLDER|-nopw|" "$WEB_DIR/scripts/start-remote-desktop.sh"
fi

chmod +x "$WEB_DIR/scripts/start-remote-desktop.sh"

# Create stop script
cat > "$WEB_DIR/scripts/stop-remote-desktop.sh" << 'STOPSCRIPT'
#!/bin/bash
# Stop Remote Desktop Access

echo "Stopping remote desktop services..."

# Stop x11vnc
pkill -f "x11vnc.*-rfbport 5900" 2>/dev/null && echo "x11vnc stopped" || echo "x11vnc was not running"

# Stop noVNC
pkill -f "websockify.*6080" 2>/dev/null && echo "noVNC stopped" || echo "noVNC was not running"

echo "Remote desktop services stopped."
STOPSCRIPT

chmod +x "$WEB_DIR/scripts/stop-remote-desktop.sh"

# Create systemd service (optional)
echo ""
echo "[4/4] Setting up systemd service (optional)..."
read -p "Create systemd service for auto-start on boot? (y/n): " CREATE_SERVICE

if [ "$CREATE_SERVICE" = "y" ]; then
    sudo tee /etc/systemd/system/field-sampler-vnc.service > /dev/null << SERVICEEOF
[Unit]
Description=Field Sampler Remote Desktop (x11vnc + noVNC)
After=display-manager.service

[Service]
Type=forking
User=$USER
Environment=DISPLAY=:0
ExecStart=$WEB_DIR/scripts/start-remote-desktop.sh
ExecStop=$WEB_DIR/scripts/stop-remote-desktop.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=graphical.target
SERVICEEOF

    sudo systemctl daemon-reload
    echo ""
    echo "Systemd service created. To enable auto-start:"
    echo "  sudo systemctl enable field-sampler-vnc"
    echo ""
    echo "To start now:"
    echo "  sudo systemctl start field-sampler-vnc"
fi

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "  To start remote desktop manually:"
echo "    $WEB_DIR/scripts/start-remote-desktop.sh"
echo ""
echo "  To stop:"
echo "    $WEB_DIR/scripts/stop-remote-desktop.sh"
echo ""
echo "  Access URL (when running):"
echo "    http://localhost:6080/vnc.html"
echo ""
echo "=============================================="
