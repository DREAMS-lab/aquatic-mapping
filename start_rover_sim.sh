#!/bin/bash
# Simple rover simulation startup - opens 3 separate terminal windows

set -e

PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_FILE="${1:-rover_fields}"
TRIAL_NUMBER="${2:-1}"

echo "================================================"
echo "  Starting R1 Rover Simulation"
echo "================================================"
echo "Launch: ${LAUNCH_FILE}.launch.py"
echo ""

# Check PX4 directory
if [ ! -d "$PX4_DIR" ]; then
    echo "ERROR: PX4 directory not found at $PX4_DIR"
    exit 1
fi

# Build ROS command
if [ "$LAUNCH_FILE" = "mission" ]; then
    ROS_CMD="ros2 launch sampling mission.launch.py trial_number:=$TRIAL_NUMBER"
else
    ROS_CMD="ros2 launch sampling ${LAUNCH_FILE}.launch.py"
fi

# Auto-detect terminal
if command -v gnome-terminal &> /dev/null; then
    TERM_CMD="gnome-terminal --"
elif command -v konsole &> /dev/null; then
    TERM_CMD="konsole -e"
elif command -v xterm &> /dev/null; then
    TERM_CMD="xterm -e"
elif command -v x-terminal-emulator &> /dev/null; then
    TERM_CMD="x-terminal-emulator -e"
else
    echo "ERROR: No terminal emulator found"
    exit 1
fi

echo "Opening 3 terminal windows..."
echo ""

# window 1: px4 sitl
$TERM_CMD bash -c "
    cd $PX4_DIR;
    make px4_sitl gz_r1_rover
" &
PID1=$!

sleep 2

# window 2: dds agent
$TERM_CMD bash -c "
    echo 'waiting 10s for px4...';
    sleep 10;
    micro-xrce-dds-agent udp4 -p 8888
" &
PID2=$!

sleep 1

# window 3: ros2 launch
$TERM_CMD bash -c "
    cd $WORKSPACE_DIR;
    source install/setup.bash;
    echo 'waiting 15s for px4 and dds...';
    sleep 15;
    $ROS_CMD
" &
PID3=$!

echo "✓ Started 3 terminals:"
echo "  1. PX4 SITL + Gazebo"
echo "  2. DDS Agent"
echo "  3. ROS2 Launch"
echo ""
echo "To stop all: ./stop_rover_sim.sh"
