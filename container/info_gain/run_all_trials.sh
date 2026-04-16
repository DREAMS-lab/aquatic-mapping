#!/bin/bash
# Run all info_gain trials for a given planner type (container-side)
# Usage: ./run_all_trials.sh <planner_type> <num_trials>
# Example: ./run_all_trials.sh exact 10
#          ./run_all_trials.sh pose_aware 10

set -e

PLANNER_TYPE="${1:-exact}"
NUM_TRIALS="${2:-10}"

WORKSPACE_DIR="${WORKSPACE_DIR:-/home/simuser/aquatic-mapping}"
PX4_DIR="${PX4_DIR:-/home/simuser/PX4-Autopilot}"
FIELDS=("radial" "x_compress" "y_compress" "x_compress_tilt" "y_compress_tilt")

# VNC ports (exact uses 5910/6090, pose_aware uses 5911/6091)
if [[ "$PLANNER_TYPE" == "exact" ]]; then
    VNC_PORT=5910
    NOVNC_PORT=6090
else
    VNC_PORT=5911
    NOVNC_PORT=6091
fi

# Status file for monitoring
STATUS_FILE="$WORKSPACE_DIR/data/trials/$PLANNER_TYPE/status.json"

# Validate planner type
if [[ "$PLANNER_TYPE" != "exact" && "$PLANNER_TYPE" != "pose_aware" ]]; then
    echo "ERROR: Invalid planner type. Use 'exact' or 'pose_aware'"
    exit 1
fi

echo "=============================================="
echo "  Info Gain Automated Trials"
echo "=============================================="
echo "  Planner:     $PLANNER_TYPE"
echo "  Num Trials:  $NUM_TRIALS"
echo "  Fields:      ${FIELDS[*]}"
echo "  Workspace:   $WORKSPACE_DIR"
echo "  Gazebo:      HEADLESS (server only)"
echo "  VNC Port:    $VNC_PORT (for RViz2/matplotlib)"
echo "  noVNC Port:  $NOVNC_PORT"
echo "=============================================="

# ============================================
# Setup virtual display (for RViz2 and matplotlib only)
# ============================================
echo "[DISPLAY] Starting Xvfb virtual display on :99..."
Xvfb :99 -screen 0 1920x1080x24 &
XVFB_PID=$!
export DISPLAY=:99
sleep 2

# Start window manager
echo "[DISPLAY] Starting fluxbox window manager..."
fluxbox &
sleep 1

# Start VNC server
echo "[VNC] Starting x11vnc on port $VNC_PORT..."
x11vnc -display :99 -forever -shared -rfbport $VNC_PORT -nopw -bg 2>/dev/null || true
sleep 1

# Start noVNC web interface
echo "[NOVNC] Starting noVNC on port $NOVNC_PORT..."
websockify --web=/usr/share/novnc/ $NOVNC_PORT localhost:$VNC_PORT &
NOVNC_PID=$!
sleep 1
echo "[NOVNC] View RViz2/matplotlib at: http://localhost:$NOVNC_PORT/vnc.html"

# ============================================
# Source ROS and activate planner venv
# ============================================
source /opt/ros/jazzy/setup.bash
source "$WORKSPACE_DIR/install/setup.bash"
source /opt/venv/$PLANNER_TYPE/bin/activate
echo "[VENV] Activated /opt/venv/$PLANNER_TYPE"

# Create directories
mkdir -p "$(dirname "$STATUS_FILE")"
mkdir -p "$WORKSPACE_DIR/data/trials/$PLANNER_TYPE/logs"

# Function to update status
update_status() {
    local trial=$1
    local field=$2
    local state=$3
    local message=$4

    cat > "$STATUS_FILE" << EOF
{
    "planner": "$PLANNER_TYPE",
    "current_trial": $trial,
    "total_trials": $NUM_TRIALS,
    "current_field": "$field",
    "state": "$state",
    "message": "$message",
    "timestamp": "$(date -Iseconds)",
    "completed_trials": $(find "$WORKSPACE_DIR/data/trials/$PLANNER_TYPE" -name "summary.json" 2>/dev/null | wc -l),
    "fields": ["radial", "x_compress", "y_compress", "x_compress_tilt", "y_compress_tilt"],
    "vnc_port": $VNC_PORT,
    "novnc_port": $NOVNC_PORT
}
EOF
}

# Initialize status
update_status 0 "" "initializing" "Starting automation..."

# Main trial loop
for trial in $(seq 1 $NUM_TRIALS); do
    for field in "${FIELDS[@]}"; do
        echo ""
        echo "========================================================"
        echo "  TRIAL $trial / $NUM_TRIALS - FIELD: $field - PLANNER: $PLANNER_TYPE"
        echo "  $(date)"
        echo "========================================================"

        TRIAL_DIR="$WORKSPACE_DIR/data/trials/$PLANNER_TYPE/$field/trial_$(printf '%03d' $trial)"

        # Check if already completed
        if [[ -f "$TRIAL_DIR/summary.json" ]]; then
            echo "[SKIP] Trial already completed: $TRIAL_DIR"
            update_status $trial $field "skipped" "Already completed"
            continue
        fi

        update_status $trial $field "starting" "Starting PX4 SITL (headless)..."

        # ============================================
        # Start PX4 SITL HEADLESS (no Gazebo GUI)
        # ============================================
        echo "[1/4] Starting PX4 SITL (headless Gazebo)..."
        cd "$PX4_DIR"
        HEADLESS=1 make px4_sitl gz_r1_rover > /tmp/px4_${PLANNER_TYPE}_${field}_${trial}.log 2>&1 &
        PX4_PID=$!
        echo "       Waiting 20s for PX4 + Gazebo server to initialize..."
        sleep 20

        # ============================================
        # Start DDS Agent
        # ============================================
        echo "[2/4] Starting DDS Agent..."
        update_status $trial $field "running" "Starting DDS Agent..."
        MicroXRCEAgent udp4 -p 8888 > /tmp/dds_${PLANNER_TYPE}_${field}_${trial}.log 2>&1 &
        DDS_PID=$!
        echo "       Waiting 5s for DDS agent..."
        sleep 5

        # ============================================
        # Start the planner (RViz2 + matplotlib will show on VNC)
        # ============================================
        echo "[3/4] Starting $PLANNER_TYPE planner for $field..."
        update_status $trial $field "running" "Launching ROS nodes..."
        cd "$WORKSPACE_DIR"

        if [[ "$PLANNER_TYPE" == "exact" ]]; then
            LAUNCH_FILE="exact.launch.py"
        else
            LAUNCH_FILE="pose_aware.launch.py"
        fi

        echo "       Waiting 15s before launching planner..."
        sleep 15

        ros2 launch info_gain $LAUNCH_FILE \
            field_type:=$field \
            trial:=$trial \
            > /tmp/planner_${PLANNER_TYPE}_${field}_${trial}.log 2>&1 &
        PLANNER_PID=$!

        update_status $trial $field "running" "Planner active - collecting samples"
        echo "[4/4] Monitoring for completion (summary.json)..."
        echo "       View RViz2: http://localhost:$NOVNC_PORT/vnc.html"

        # Wait for completion - NO TIMEOUT
        while true; do
            # Check if summary.json exists (mission complete)
            if [[ -f "$TRIAL_DIR/summary.json" ]]; then
                echo "[COMPLETE] Trial finished successfully!"
                echo "  Results: $TRIAL_DIR"
                update_status $trial $field "completed" "Trial completed successfully"
                break
            fi

            # Check if planner crashed
            if ! kill -0 $PLANNER_PID 2>/dev/null; then
                echo "[ERROR] Planner process died unexpectedly"
                echo "       Check log: /tmp/planner_${PLANNER_TYPE}_${field}_${trial}.log"
                echo "--- Last 30 lines of planner log ---"
                tail -30 /tmp/planner_${PLANNER_TYPE}_${field}_${trial}.log 2>/dev/null || true
                echo "---"
                update_status $trial $field "error" "Planner crashed"
                break
            fi

            # Update status with sample count if available
            if [[ -f "$TRIAL_DIR/samples.csv" ]]; then
                SAMPLE_COUNT=$(wc -l < "$TRIAL_DIR/samples.csv")
                SAMPLE_COUNT=$((SAMPLE_COUNT - 1))  # Subtract header
                update_status $trial $field "running" "Collecting samples: $SAMPLE_COUNT / 100"
            fi

            sleep 10
        done

        # Stop all processes for this trial
        echo "[CLEANUP] Stopping processes..."
        kill $PLANNER_PID 2>/dev/null || true
        kill $DDS_PID 2>/dev/null || true
        kill $PX4_PID 2>/dev/null || true
        pkill -f "gz sim" 2>/dev/null || true
        pkill -f "ruby" 2>/dev/null || true

        # Wait for cleanup
        sleep 5

        # Copy logs to trial directory
        mkdir -p "$TRIAL_DIR"
        cp /tmp/px4_${PLANNER_TYPE}_${field}_${trial}.log "$TRIAL_DIR/" 2>/dev/null || true
        cp /tmp/dds_${PLANNER_TYPE}_${field}_${trial}.log "$TRIAL_DIR/" 2>/dev/null || true
        cp /tmp/planner_${PLANNER_TYPE}_${field}_${trial}.log "$TRIAL_DIR/" 2>/dev/null || true

        echo "[DONE] Field $field trial $trial complete"
        echo ""
    done
done

update_status $NUM_TRIALS "" "finished" "All trials complete"

# Cleanup display
kill $NOVNC_PID 2>/dev/null || true
kill $XVFB_PID 2>/dev/null || true

echo "=============================================="
echo "  ALL TRIALS COMPLETE"
echo "  $(date)"
echo "=============================================="
