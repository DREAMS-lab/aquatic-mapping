#!/bin/bash
# Aquatic Mapping Simulation Container Entrypoint
#
# Environment variables:
#   TRIAL_ID      - Trial number (default: 1)
#   FIELD_TYPE    - Field type: radial, x_compress, y_compress, etc. (default: radial)
#   ROS_DOMAIN_ID - ROS2 domain for isolation (default: 0)
#   HEADLESS      - 1 for headless, 0 for GUI (default: 1)
#   VNC_PORT      - VNC server port (default: 5900)
#   NOVNC_PORT    - noVNC web port (default: 6080)
#   PX4_MODEL     - PX4 vehicle model: gz_r1_rover, gz_x500, etc. (default: gz_r1_rover)

set -e

# Defaults
TRIAL_ID="${TRIAL_ID:-1}"
FIELD_TYPE="${FIELD_TYPE:-radial}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
HEADLESS="${HEADLESS:-1}"
VNC_PORT="${VNC_PORT:-5900}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
PX4_MODEL="${PX4_MODEL:-gz_r1_rover}"

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

# Export ROS_DOMAIN_ID for isolation
export ROS_DOMAIN_ID

# Source ROS2
source /opt/ros/jazzy/setup.bash

# Source workspace
if [ -f /home/simuser/aquatic-mapping/install/setup.bash ]; then
    source /home/simuser/aquatic-mapping/install/setup.bash
fi

# Function to start virtual display
start_display() {
    echo "[DISPLAY] Starting Xvfb virtual display on :99..."
    Xvfb :99 -screen 0 1920x1080x24 &
    export DISPLAY=:99
    sleep 2

    # Start a minimal window manager
    fluxbox &
    sleep 1
}

# Function to start VNC server
start_vnc() {
    echo "[VNC] Starting x11vnc on port $VNC_PORT..."
    x11vnc -display :99 -forever -shared -rfbport $VNC_PORT -nopw -bg
    sleep 1
}

# Function to start noVNC web interface
start_novnc() {
    echo "[NOVNC] Starting noVNC web interface on port $NOVNC_PORT..."
    websockify --web=/usr/share/novnc/ $NOVNC_PORT localhost:$VNC_PORT &
    sleep 1
    echo "[NOVNC] =================================================="
    echo "[NOVNC] Web interface available at:"
    echo "[NOVNC]   Local:  http://localhost:$NOVNC_PORT/vnc.html"
    echo "[NOVNC]   Remote: http://<your-ip>:$NOVNC_PORT/vnc.html"
    echo "[NOVNC] =================================================="
}

# Function to start Micro XRCE DDS Agent
start_xrce_agent() {
    echo "[XRCE] Starting Micro XRCE DDS Agent (same as snap --edge)..."
    micro-xrce-dds-agent udp4 -p 8888 &
    XRCE_PID=$!
    sleep 2
    echo "[XRCE] Agent started with PID $XRCE_PID"
}

# Function to start PX4 SITL
start_px4() {
    echo "[PX4] Starting PX4 SITL with model: $PX4_MODEL..."
    cd /home/simuser/PX4-Autopilot

    if [ "$HEADLESS" = "1" ]; then
        echo "[PX4] Running in HEADLESS mode (no Gazebo GUI)..."
        export HEADLESS=1
    else
        echo "[PX4] Running with GUI..."
    fi

    # Run PX4 with the specified model
    make px4_sitl $PX4_MODEL &
    PX4_PID=$!

    sleep 15  # Give PX4 time to initialize
    echo "[PX4] Started with PID $PX4_PID"
}

# Function to start the mission
start_mission() {
    echo "[MISSION] Waiting for PX4 and ROS2 to be ready..."
    sleep 5

    echo "[MISSION] Starting mission: $FIELD_TYPE trial $TRIAL_ID"
    cd /home/simuser/aquatic-mapping

    # Launch the mission
    ros2 launch sampling mission.launch.py trial_number:=$TRIAL_ID &
    MISSION_PID=$!

    echo "[MISSION] Mission launched with PID $MISSION_PID"

    # Wait for mission to complete (or run indefinitely)
    wait $MISSION_PID
}

# Function to run only the rover (no mission)
start_rover_only() {
    echo "[ROVER] Starting rover stack only..."
    cd /home/simuser/aquatic-mapping
    ros2 launch sampling rover_fields.launch.py &
}

# Cleanup function
cleanup() {
    echo "[CLEANUP] Shutting down..."
    pkill -f "px4" || true
    pkill -f "gz sim" || true
    pkill -f "MicroXRCEAgent" || true
    pkill -f "micro-xrce-dds-agent" || true
    pkill -f "ros2" || true
    pkill -f "x11vnc" || true
    pkill -f "Xvfb" || true
    echo "[CLEANUP] Done."
    exit 0
}

trap cleanup SIGTERM SIGINT

# Main execution based on command
case "${1:-mission}" in
    mission)
        # Full mission execution
        start_display
        start_vnc
        start_novnc
        start_xrce_agent
        start_px4
        start_mission
        ;;

    rover)
        # Just start rover stack (for debugging)
        start_display
        start_vnc
        start_novnc
        start_xrce_agent
        start_px4
        start_rover_only
        # Keep container running
        tail -f /dev/null
        ;;

    shell)
        # Interactive shell
        start_display
        start_vnc
        start_novnc
        exec /bin/bash
        ;;

    headless-test)
        # Minimal test without display
        echo "[TEST] Running headless test..."
        start_xrce_agent
        start_px4
        sleep 30
        echo "[TEST] Test complete."
        ;;

    *)
        # Custom command
        exec "$@"
        ;;
esac

# Keep container alive
wait
