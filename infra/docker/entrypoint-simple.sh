#!/bin/bash
# Simple entrypoint that uses native startup scripts

set -e

TRIAL_ID="${TRIAL_ID:-1}"
FIELD_TYPE="${FIELD_TYPE:-radial}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
HEADLESS="${HEADLESS:-1}"
VNC_PORT="${VNC_PORT:-5900}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
PX4_MODEL="${PX4_MODEL:-gz_r1_rover}"

# Ensure workspace path is set for data recording
export AQUATIC_WORKSPACE="${AQUATIC_WORKSPACE:-/home/simuser/aquatic-mapping}"

# Create data directory structure (may already exist from host mount)
DATA_DIR="$AQUATIC_WORKSPACE/src/sampling/data/missions/trial_${TRIAL_ID}"
mkdir -p "$DATA_DIR" 2>/dev/null || echo "[DATA] Using existing directory (mounted from host)"
echo "[DATA] Output directory: $DATA_DIR"

# Ensure we can write to the data directory
if ! touch "$DATA_DIR/.write_test" 2>/dev/null; then
    echo "[WARNING] Cannot write to $DATA_DIR - data may not be saved!"
else
    rm -f "$DATA_DIR/.write_test"
    echo "[DATA] Write access confirmed"
fi

echo "=============================================="
echo "  Aquatic Mapping Simulation Container"
echo "=============================================="
echo "  Trial ID:      $TRIAL_ID"
echo "  Field Type:    $FIELD_TYPE"
echo "  ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "  Headless:      $HEADLESS"
echo "  VNC Port:      $VNC_PORT"
echo "  noVNC Port:    $NOVNC_PORT"
echo "  PX4 Model:     $PX4_MODEL"
echo "=============================================="

export ROS_DOMAIN_ID

# Verify DDS agent is available
if ! command -v MicroXRCEAgent &> /dev/null; then
    echo "[ERROR] MicroXRCEAgent not found in PATH"
    exit 1
fi
echo "[DDS] MicroXRCEAgent ready"

# Start virtual display
echo "[DISPLAY] Starting Xvfb virtual display on :99..."
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99
sleep 2

# Start window manager
fluxbox &
sleep 1

# Start VNC server
echo "[VNC] Starting x11vnc on port $VNC_PORT..."
x11vnc -display :99 -forever -shared -rfbport $VNC_PORT -nopw -bg
sleep 1

# Start noVNC web interface
echo "[NOVNC] Starting noVNC web interface on port $NOVNC_PORT..."
websockify --web=/usr/share/novnc/ $NOVNC_PORT localhost:$VNC_PORT &
sleep 1
echo "[NOVNC] Web interface available at: http://<your-ip>:$NOVNC_PORT/vnc.html"

# Run the command passed in (mission/rover/etc)
case "${1:-mission}" in
    mission)
        echo "[STARTUP] Running headless mission..."
        /bin/bash /home/simuser/start-mission-headless.sh $TRIAL_ID
        ;;

    rover)
        echo "[STARTUP] Running rover with native script..."
        cd /home/simuser/aquatic-mapping
        /bin/bash /home/simuser/aquatic-mapping/start_rover_sim.sh rover_fields
        # Keep running
        wait
        ;;

    shell)
        echo "[SHELL] Starting interactive shell..."
        exec /bin/bash
        ;;

    *)
        exec "$@"
        ;;
esac
