#!/bin/bash
# Headless mission startup for containers
# Runs PX4, DDS agent, and ROS2 mission in background

set -e

PX4_DIR="${PX4_DIR:-/home/simuser/PX4-Autopilot}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/home/simuser/aquatic-mapping}"
TRIAL_NUMBER="${1:-1}"

echo "================================================"
echo "  Starting Headless Mission (Trial $TRIAL_NUMBER)"
echo "================================================"

# Start PX4 SITL in background
echo "[1/3] Starting PX4 SITL + Gazebo..."
cd "$PX4_DIR"
nohup make px4_sitl gz_r1_rover > /tmp/px4.log 2>&1 &
PX4_PID=$!
echo "  PX4 PID: $PX4_PID"

# Wait for PX4 to initialize
echo "  Waiting 15s for PX4 to initialize..."
sleep 15

# Start DDS agent in background
echo "[2/3] Starting MicroXRCEAgent..."
nohup MicroXRCEAgent udp4 -p 8888 > /tmp/dds.log 2>&1 &
DDS_PID=$!
echo "  DDS PID: $DDS_PID"

# Wait for DDS to connect
echo "  Waiting 5s for DDS connection..."
sleep 5

# Start ROS2 mission in background
echo "[3/3] Starting ROS2 mission launch..."
cd "$WORKSPACE_DIR"
source /opt/ros/jazzy/setup.bash
source install/setup.bash
nohup ros2 launch sampling mission.launch.py trial_number:=$TRIAL_NUMBER > /tmp/ros2.log 2>&1 &
ROS2_PID=$!
echo "  ROS2 PID: $ROS2_PID"

echo ""
echo "✓ All processes started!"
echo "  PX4:  $PX4_PID (log: /tmp/px4.log)"
echo "  DDS:  $DDS_PID (log: /tmp/dds.log)"
echo "  ROS2: $ROS2_PID (log: /tmp/ros2.log)"
echo ""
echo "Monitor logs:"
echo "  tail -f /tmp/px4.log"
echo "  tail -f /tmp/dds.log"
echo "  tail -f /tmp/ros2.log"
echo ""

# Start mission completion monitor
echo "[4/4] Starting mission completion monitor..."
/bin/bash /home/simuser/mission-monitor.sh $TRIAL_NUMBER > /tmp/monitor.log 2>&1 &
MONITOR_PID=$!
echo "  Monitor PID: $MONITOR_PID"
echo ""

# Keep script running (wait for all background processes)
wait
