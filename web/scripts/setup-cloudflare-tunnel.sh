#!/bin/bash
# =============================================================================
# Cloudflare Tunnel Setup Script for Field Sampler
# Securely expose your local server to bharathdesikan.com
# =============================================================================

set -e

echo "=============================================="
echo "  Field Sampler - Cloudflare Tunnel Setup"
echo "=============================================="
echo ""

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "[1/4] Installing cloudflared..."
    # Download and install cloudflared
    curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    sudo dpkg -i cloudflared.deb
    rm cloudflared.deb
    echo "cloudflared installed successfully"
else
    echo "[1/4] cloudflared already installed"
fi

echo ""
echo "[2/4] Cloudflare Login"
echo "=============================================="
echo ""
echo "You need to authenticate with Cloudflare."
echo "This will open a browser window to log in."
echo ""
read -p "Press Enter to continue..."

cloudflared tunnel login

echo ""
echo "[3/4] Create Tunnel"
echo "=============================================="
echo ""
read -p "Enter a name for your tunnel (e.g., field-sampler): " TUNNEL_NAME

# Create the tunnel
cloudflared tunnel create "$TUNNEL_NAME"

# Get tunnel ID
TUNNEL_ID=$(cloudflared tunnel list | grep "$TUNNEL_NAME" | awk '{print $1}')
echo ""
echo "Tunnel created with ID: $TUNNEL_ID"

echo ""
echo "[4/4] Configure DNS"
echo "=============================================="
echo ""
echo "Now let's set up DNS routing for your domain."
echo ""
read -p "Enter your domain (e.g., bharathdesikan.com): " DOMAIN
read -p "Enter subdomain for Field Sampler (leave empty for root, or enter e.g., 'app'): " SUBDOMAIN

if [ -z "$SUBDOMAIN" ]; then
    HOSTNAME="$DOMAIN"
else
    HOSTNAME="$SUBDOMAIN.$DOMAIN"
fi

# Create DNS route
cloudflared tunnel route dns "$TUNNEL_NAME" "$HOSTNAME"

# Create config file
CONFIG_DIR="$HOME/.cloudflared"
mkdir -p "$CONFIG_DIR"

cat > "$CONFIG_DIR/config.yml" << EOF
tunnel: $TUNNEL_ID
credentials-file: $CONFIG_DIR/$TUNNEL_ID.json

ingress:
  # Main website and dashboard
  - hostname: $HOSTNAME
    service: http://localhost:8000

  # noVNC remote desktop (optional subdomain)
  - hostname: vnc.$DOMAIN
    service: http://localhost:6080

  # Catch-all rule
  - service: http_status:404
EOF

echo ""
echo "Configuration saved to $CONFIG_DIR/config.yml"

# Create systemd service
echo ""
read -p "Create systemd service for auto-start? (y/n): " CREATE_SERVICE

if [ "$CREATE_SERVICE" = "y" ]; then
    sudo cloudflared service install
    sudo systemctl enable cloudflared
    echo ""
    echo "Systemd service created and enabled."
    echo "To start: sudo systemctl start cloudflared"
fi

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "  Your tunnel is configured for: $HOSTNAME"
echo ""
echo "  To start the tunnel manually:"
echo "    cloudflared tunnel run $TUNNEL_NAME"
echo ""
echo "  To start as a service:"
echo "    sudo systemctl start cloudflared"
echo ""
echo "  IMPORTANT: Make sure the Field Sampler server"
echo "  is running on http://localhost:8000"
echo ""
echo "  Start the server with:"
echo "    cd ~/workspaces/aquatic-mapping/web"
echo "    ./start.sh"
echo ""
echo "=============================================="
echo ""
echo "Security Notes:"
echo "  - All traffic is encrypted via Cloudflare"
echo "  - Dashboard requires authentication (bakin/ozhugu)"
echo "  - Change default password in start.sh or via env vars"
echo "  - Consider enabling Cloudflare Access for extra security"
echo ""
