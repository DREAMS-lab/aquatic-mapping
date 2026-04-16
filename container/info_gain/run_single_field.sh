#!/bin/bash
# Container-side script: Run ONE field for ONE planner, then exit
# This script is designed to be called by the host orchestrator
#
# Usage: ./run_single_field.sh <planner_type> <field_type> <trial_num> <slot>
# Example: ./run_single_field.sh exact radial 1 0

set -e

PLANNER_TYPE="${1:-exact}"
FIELD_TYPE="${2:-radial}"
TRIAL_NUM="${3:-1}"
SLOT="${4:-0}"  # Worker slot for parallel mode

WORKSPACE_DIR="${WORKSPACE_DIR:-/home/simuser/aquatic-mapping}"
PX4_DIR="${PX4_DIR:-/home/simuser/PX4-Autopilot}"

# Analytical and nonstationary planners use same venv as exact (same dependencies)
if [[ "$PLANNER_TYPE" == "analytical" || "$PLANNER_TYPE" == "nonstationary_exact" || "$PLANNER_TYPE" == "nonstationary_pose_aware" || "$PLANNER_TYPE" == "nonstationary_hotspot_exact" || "$PLANNER_TYPE" == "nonstationary_hotspot_pose_aware" ]]; then
    VENV_PATH="/opt/venv/exact"
else
    VENV_PATH="/opt/venv/${PLANNER_TYPE}"
fi

# Output directory - all planners save to centralized data/trials/
TRIAL_DIR="$WORKSPACE_DIR/data/trials/$PLANNER_TYPE/$FIELD_TYPE/trial_$(printf '%03d' $TRIAL_NUM)"
LOG_DIR="$TRIAL_DIR"
COMPUTE_LOG="$TRIAL_DIR/compute_metrics.csv"

echo "=============================================="
echo "  Single Field Runner"
echo "=============================================="
echo "  Planner:    $PLANNER_TYPE"
echo "  Field:      $FIELD_TYPE"
echo "  Trial:      $TRIAL_NUM"
echo "  Slot:       $SLOT"
echo "  Output:     $TRIAL_DIR"
echo "  Workspace:  $WORKSPACE_DIR"
echo "=============================================="

# Create output directory
mkdir -p "$TRIAL_DIR"
mkdir -p "$LOG_DIR"

# ============================================
# PHASE 1: Setup virtual display + VNC
# ============================================
echo "[1/7] Setting up virtual display and VNC..."

# VNC ports are per-slot (workers can run any planner)
# Slot 0: VNC 5902, noVNC 6090
# Slot 1: VNC 5903, noVNC 6091
# Slot 2: VNC 5904, noVNC 6092
# Slot 3: VNC 5905, noVNC 6093
# Slot 4: VNC 5906, noVNC 6094
# Slot 5: VNC 5907, noVNC 6095
case "$SLOT" in
    0) VNC_PORT=5902; NOVNC_PORT=6090 ;;
    1) VNC_PORT=5903; NOVNC_PORT=6091 ;;
    2) VNC_PORT=5904; NOVNC_PORT=6092 ;;
    3) VNC_PORT=5905; NOVNC_PORT=6093 ;;
    4) VNC_PORT=5906; NOVNC_PORT=6094 ;;
    5) VNC_PORT=5907; NOVNC_PORT=6095 ;;
    *) VNC_PORT=$((5902 + SLOT)); NOVNC_PORT=$((6090 + SLOT)) ;;
esac

# Use unique display number per slot
DISPLAY_NUM=$((99 + SLOT))

# ============================================
# CRITICAL: Isolate ROS2 and Gazebo per slot
# ============================================
# Without this, containers on the same Docker network will
# discover each other's ROS2 topics and Gazebo worlds,
# causing planners to interfere with each other!
export ROS_DOMAIN_ID=$((SLOT + 10))
export GZ_PARTITION="worker_${SLOT}"
echo "       ROS_DOMAIN_ID=$ROS_DOMAIN_ID (isolated per slot)"
echo "       GZ_PARTITION=$GZ_PARTITION (isolated Gazebo world)"

# Start virtual display
Xvfb :$DISPLAY_NUM -screen 0 1920x1080x24 &
XVFB_PID=$!
export DISPLAY=:$DISPLAY_NUM
sleep 2

# Start window manager
fluxbox &
sleep 1

# Start VNC server
echo "       Starting x11vnc on port $VNC_PORT (display :$DISPLAY_NUM)..."
x11vnc -display :$DISPLAY_NUM -forever -shared -rfbport $VNC_PORT -nopw -bg 2>/dev/null || true
sleep 1

# Start noVNC web interface
echo "       Starting noVNC on port $NOVNC_PORT..."
websockify --web=/usr/share/novnc/ $NOVNC_PORT localhost:$VNC_PORT &
sleep 1
echo "       VNC available at: http://localhost:$NOVNC_PORT/vnc.html"

# ============================================
# PHASE 2: Source ROS and build workspace
# ============================================
echo "[2/7] Sourcing ROS and building workspace..."
source /opt/ros/jazzy/setup.bash

cd "$WORKSPACE_DIR"

# Build with --symlink-install so Python scripts are symlinked (not copied) from source.
# This guarantees the install/ always reflects the latest source — no stale cache issues.
# Use a file lock to prevent concurrent containers from corrupting shared build/install dirs.
LOCK_FILE="$WORKSPACE_DIR/.build.lock"

echo "       Waiting for build lock..."
(
    flock -x 200

    # Fix CMakeCache path mismatch (host vs container paths)
    for pkg in px4_msgs info_gain sampling nonstationary_planning; do
        if [ -f "build/$pkg/CMakeCache.txt" ]; then
            CACHED_DIR=$(grep "CMAKE_HOME_DIRECTORY" "build/$pkg/CMakeCache.txt" 2>/dev/null | cut -d= -f2 || true)
            if [ -n "$CACHED_DIR" ] && [ "$CACHED_DIR" != "$(pwd)" ]; then
                echo "       Clearing stale $pkg CMake cache (wrong paths)..."
                rm -rf "build/$pkg" "install/$pkg"
            fi
        fi
    done

    echo "       Building workspace..."
    colcon build --symlink-install --packages-select px4_msgs info_gain sampling nonstationary_planning 2>&1 | tee "$LOG_DIR/build.log" || {
        echo "[ERROR] Build failed! Check $LOG_DIR/build.log"
        exit 1
    }

) 200>"$LOCK_FILE"

source "$WORKSPACE_DIR/install/setup.bash"

# Add venv packages to PYTHONPATH (don't activate - keeps ROS2 packages accessible)
if [[ -d "$VENV_PATH" ]]; then
    export PYTHONPATH="$VENV_PATH/lib/python3.12/site-packages:$PYTHONPATH"
    echo "       Added venv to PYTHONPATH: $VENV_PATH"
else
    echo "[WARNING] Virtual environment not found: $VENV_PATH"
fi

# ============================================
# PHASE 3: Start PX4 SITL (headless)
# ============================================
echo "[3/7] Starting PX4 SITL (headless Gazebo)..."
cd "$PX4_DIR"
HEADLESS=1 make px4_sitl gz_r1_rover > /dev/null 2>&1 &
PX4_PID=$!

# Wait for Gazebo to be ready (check for gz topics)
echo "       Waiting for Gazebo to initialize..."
MAX_WAIT=60
WAITED=0
while ! gz topic -l 2>/dev/null | grep -q "/world"; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        echo "[ERROR] Gazebo failed to start within ${MAX_WAIT}s"
        exit 1
    fi
    echo "       Waiting... ($WAITED/${MAX_WAIT}s)"
done
echo "       Gazebo ready after ${WAITED}s"

# ============================================
# PHASE 4: Start DDS Agent
# ============================================
echo "[4/7] Starting MicroXRCE-DDS Agent..."
MicroXRCEAgent udp4 -p 8888 > "$LOG_DIR/dds.log" 2>&1 &
DDS_PID=$!

# Wait for DDS bridge (check for PX4 topics in ROS2)
echo "       Waiting for DDS bridge..."
MAX_WAIT=30
WAITED=0
cd "$WORKSPACE_DIR"
source "$WORKSPACE_DIR/install/setup.bash"
while ! ros2 topic list 2>/dev/null | grep -q "/fmu/out"; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        echo "[ERROR] DDS bridge failed to start within ${MAX_WAIT}s"
        cat "$LOG_DIR/dds.log" | tail -20
        exit 1
    fi
    echo "       Waiting... ($WAITED/${MAX_WAIT}s)"
done
echo "       DDS bridge ready after ${WAITED}s"

# ============================================
# PHASE 5: Start compute logger (background)
# ============================================
echo "[5/7] Starting compute metrics logger..."
(
    # CSV header
    echo "timestamp,cpu_percent,ram_used_mb,ram_total_mb,gpu_util,gpu_mem_used_mb,gpu_mem_total_mb,gpu_temp" > "$COMPUTE_LOG"

    while true; do
        TIMESTAMP=$(date +%s.%N)

        # CPU usage (average across cores)
        CPU_PERCENT=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)

        # RAM usage
        RAM_INFO=$(free -m | grep Mem)
        RAM_TOTAL=$(echo $RAM_INFO | awk '{print $2}')
        RAM_USED=$(echo $RAM_INFO | awk '{print $3}')

        # GPU usage (nvidia-smi)
        if command -v nvidia-smi &> /dev/null; then
            GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "0,0,0,0")
            GPU_UTIL=$(echo $GPU_INFO | cut -d',' -f1 | tr -d ' ')
            GPU_MEM_USED=$(echo $GPU_INFO | cut -d',' -f2 | tr -d ' ')
            GPU_MEM_TOTAL=$(echo $GPU_INFO | cut -d',' -f3 | tr -d ' ')
            GPU_TEMP=$(echo $GPU_INFO | cut -d',' -f4 | tr -d ' ')
        else
            GPU_UTIL=0
            GPU_MEM_USED=0
            GPU_MEM_TOTAL=0
            GPU_TEMP=0
        fi

        echo "$TIMESTAMP,$CPU_PERCENT,$RAM_USED,$RAM_TOTAL,$GPU_UTIL,$GPU_MEM_USED,$GPU_MEM_TOTAL,$GPU_TEMP" >> "$COMPUTE_LOG"

        sleep 1
    done
) &
COMPUTE_PID=$!
echo "       Compute logger started (PID: $COMPUTE_PID)"

# ============================================
# PHASE 6: Launch ROS2 planner
# ============================================
echo "[6/7] Launching $PLANNER_TYPE planner for $FIELD_TYPE..."

if [[ "$PLANNER_TYPE" == "exact" ]]; then
    LAUNCH_PKG="info_gain"
    LAUNCH_FILE="exact.launch.py"
elif [[ "$PLANNER_TYPE" == "analytical" ]]; then
    LAUNCH_PKG="info_gain"
    LAUNCH_FILE="analytical.launch.py"
elif [[ "$PLANNER_TYPE" == "nonstationary_exact" ]]; then
    LAUNCH_PKG="nonstationary_planning"
    LAUNCH_FILE="nonstationary_exact.launch.py"
elif [[ "$PLANNER_TYPE" == "nonstationary_pose_aware" ]]; then
    LAUNCH_PKG="nonstationary_planning"
    LAUNCH_FILE="nonstationary_pose_aware.launch.py"
elif [[ "$PLANNER_TYPE" == "nonstationary_hotspot_exact" ]]; then
    LAUNCH_PKG="nonstationary_planning"
    LAUNCH_FILE="nonstationary_hotspot_exact.launch.py"
elif [[ "$PLANNER_TYPE" == "nonstationary_hotspot_pose_aware" ]]; then
    LAUNCH_PKG="nonstationary_planning"
    LAUNCH_FILE="nonstationary_hotspot_pose_aware.launch.py"
else
    LAUNCH_PKG="info_gain"
    LAUNCH_FILE="pose_aware.launch.py"
fi

cd "$WORKSPACE_DIR"

# Build launch args — always include field_type and trial
LAUNCH_ARGS="field_type:=$FIELD_TYPE trial:=$TRIAL_NUM"

# Optional: pass uncertainty_scale if set via env var
if [ -n "$UNCERTAINTY_SCALE" ]; then
    LAUNCH_ARGS="$LAUNCH_ARGS uncertainty_scale:=$UNCERTAINTY_SCALE"
fi

ros2 launch $LAUNCH_PKG $LAUNCH_FILE $LAUNCH_ARGS \
    > "$LOG_DIR/planner.log" 2>&1 &
PLANNER_PID=$!

echo "       Planner started (PID: $PLANNER_PID)"
echo "       Logs: $LOG_DIR/planner.log"

# ============================================
# PHASE 7: Wait for completion
# ============================================
echo "[7/7] Waiting for mission completion (summary.json)..."
echo "       Checking every 10s..."

START_TIME=$(date +%s)
while true; do
    # Check if summary.json exists (mission complete)
    if [[ -f "$TRIAL_DIR/summary.json" ]]; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo ""
        echo "[COMPLETE] Mission finished in ${DURATION}s"
        echo "  Results: $TRIAL_DIR"
        break
    fi

    # Check if planner crashed
    if ! kill -0 $PLANNER_PID 2>/dev/null; then
        echo ""
        echo "[ERROR] Planner process died unexpectedly!"
        echo "--- Last 50 lines of planner log ---"
        tail -50 "$LOG_DIR/planner.log" 2>/dev/null || true
        echo "---"
        break
    fi

    # Show progress if samples.csv exists
    if [[ -f "$TRIAL_DIR/samples.csv" ]]; then
        SAMPLE_COUNT=$(wc -l < "$TRIAL_DIR/samples.csv")
        SAMPLE_COUNT=$((SAMPLE_COUNT - 1))  # Subtract header
        ELAPSED=$(($(date +%s) - START_TIME))
        echo "  [$ELAPSED s] Samples collected: $SAMPLE_COUNT / 100"
    fi

    sleep 10
done

# ============================================
# CLEANUP
# ============================================
echo ""
echo "[CLEANUP] Stopping processes..."

# Stop compute logger
kill $COMPUTE_PID 2>/dev/null || true

# Stop planner
kill $PLANNER_PID 2>/dev/null || true
sleep 2

# Stop DDS
kill $DDS_PID 2>/dev/null || true

# Stop PX4/Gazebo
kill $PX4_PID 2>/dev/null || true
pkill -f "gz sim" 2>/dev/null || true
pkill -f "ruby" 2>/dev/null || true

# Stop display
kill $XVFB_PID 2>/dev/null || true

# Wait for cleanup
sleep 3

# Generate compute summary
if [[ -f "$COMPUTE_LOG" ]]; then
    echo "[COMPUTE] Generating metrics summary..."
    python3 << EOF
import csv
import json
import sys

metrics = {'cpu': [], 'ram': [], 'gpu_util': [], 'gpu_mem': [], 'gpu_temp': []}

try:
    with open('$COMPUTE_LOG', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics['cpu'].append(float(row['cpu_percent']))
            metrics['ram'].append(float(row['ram_used_mb']))
            metrics['gpu_util'].append(float(row['gpu_util']))
            metrics['gpu_mem'].append(float(row['gpu_mem_used_mb']))
            metrics['gpu_temp'].append(float(row['gpu_temp']))

    summary = {
        'cpu_mean': sum(metrics['cpu']) / len(metrics['cpu']) if metrics['cpu'] else 0,
        'cpu_max': max(metrics['cpu']) if metrics['cpu'] else 0,
        'ram_mean_mb': sum(metrics['ram']) / len(metrics['ram']) if metrics['ram'] else 0,
        'ram_max_mb': max(metrics['ram']) if metrics['ram'] else 0,
        'gpu_util_mean': sum(metrics['gpu_util']) / len(metrics['gpu_util']) if metrics['gpu_util'] else 0,
        'gpu_util_max': max(metrics['gpu_util']) if metrics['gpu_util'] else 0,
        'gpu_mem_mean_mb': sum(metrics['gpu_mem']) / len(metrics['gpu_mem']) if metrics['gpu_mem'] else 0,
        'gpu_mem_max_mb': max(metrics['gpu_mem']) if metrics['gpu_mem'] else 0,
        'gpu_temp_max': max(metrics['gpu_temp']) if metrics['gpu_temp'] else 0,
        'samples': len(metrics['cpu'])
    }

    with open('$TRIAL_DIR/compute_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  CPU: {summary['cpu_mean']:.1f}% avg, {summary['cpu_max']:.1f}% max")
    print(f"  RAM: {summary['ram_mean_mb']:.0f}MB avg, {summary['ram_max_mb']:.0f}MB max")
    print(f"  GPU: {summary['gpu_util_mean']:.1f}% avg, {summary['gpu_mem_max_mb']:.0f}MB max mem")
except Exception as e:
    print(f"  Error generating summary: {e}", file=sys.stderr)
EOF
fi

# Fix ownership so host user can manage files
# Get host UID/GID from the mounted directory's parent
HOST_UID=$(stat -c '%u' "$WORKSPACE_DIR")
HOST_GID=$(stat -c '%g' "$WORKSPACE_DIR")
echo "Fixing file ownership to $HOST_UID:$HOST_GID..."
chown -R "$HOST_UID:$HOST_GID" "$TRIAL_DIR" 2>&1 | head -5 || echo "Warning: Some files may still be root-owned"

echo ""
echo "=============================================="
echo "  FIELD COMPLETE: $PLANNER_TYPE / $FIELD_TYPE / trial $TRIAL_NUM (slot $SLOT)"
echo "=============================================="
echo ""

exit 0
