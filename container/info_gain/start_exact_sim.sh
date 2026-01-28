#!/bin/bash
# Start exact planner simulation (no uncertainty consideration)
# Usage: ./start_exact_sim.sh [field_type] [trial]
# Example: ./start_exact_sim.sh radial 1

set -e

PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIELD_TYPE="${1:-radial}"
TRIAL="${2:--1}"

# Parameters (override via environment)
HORIZON="${HORIZON:-2}"
LAMBDA_COST="${LAMBDA_COST:-0.1}"

echo "================================================"
echo "  Exact Planner Simulation (No Uncertainty)"
echo "================================================"
echo "Workspace:  $WORKSPACE_DIR"
echo "Field:      $FIELD_TYPE"
if [ "$TRIAL" -ge 0 ] 2>/dev/null; then
    echo "Trial:      $TRIAL"
else
    echo "Trial:      auto (next available)"
fi
echo "Horizon:    $HORIZON"
echo "Lambda:     $LAMBDA_COST"
echo "Max samples: 100 (hardcoded)"
echo ""

# Check PX4 directory
if [ ! -d "$PX4_DIR" ]; then
    echo "ERROR: PX4 directory not found at $PX4_DIR"
    echo "Set PX4_DIR environment variable"
    exit 1
fi

# Auto-detect terminal
if command -v gnome-terminal &> /dev/null; then
    TERM_CMD="gnome-terminal --"
elif command -v konsole &> /dev/null; then
    TERM_CMD="konsole -e"
elif command -v xterm &> /dev/null; then
    TERM_CMD="xterm -e"
else
    echo "ERROR: No terminal emulator found"
    exit 1
fi

echo "Opening 3 terminal windows..."
echo ""

# Window 1: PX4 SITL + Gazebo
echo "[1/3] Starting PX4 SITL + Gazebo..."
$TERM_CMD bash -c "
    cd $PX4_DIR
    echo '=== PX4 SITL + Gazebo ==='
    make px4_sitl gz_r1_rover
" &
sleep 2

# Window 2: DDS Agent
echo "[2/3] Starting DDS Agent..."
$TERM_CMD bash -c "
    echo '=== DDS Agent ==='
    echo 'Waiting 15s for PX4...'
    sleep 15
    MicroXRCEAgent udp4 -p 8888
" &
sleep 1

# Window 3: Launch file (TF, field, RViz, exact planner)
echo "[3/3] Starting ROS2 launch..."
$TERM_CMD bash -c "
    cd $WORKSPACE_DIR
    source /opt/ros/jazzy/setup.bash
    source install/setup.bash
    echo '=== ROS2 Launch (Exact Planner) ==='
    echo 'Waiting 20s for PX4 and DDS...'
    sleep 20
    ros2 launch info_gain exact.launch.py \
        field_type:=$FIELD_TYPE \
        trial:=$TRIAL \
        horizon:=$HORIZON \
        lambda_cost:=$LAMBDA_COST
" &

echo ""
echo "================================================"
echo "  Started 3 terminals:"
echo "    1. PX4 SITL + Gazebo"
echo "    2. DDS Agent"
echo "    3. ROS2 Launch (Exact Planner - No Uncertainty)"
echo ""
echo "  Data saved to:"
echo "    $WORKSPACE_DIR/src/info_gain/data/trials/exact/$FIELD_TYPE/trial_NNN/"
echo ""
echo "  To stop: ./stop_sim.sh"
echo "================================================"
