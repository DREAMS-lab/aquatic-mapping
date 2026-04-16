#!/bin/bash
# Host-side orchestrator for running info_gain trials in Docker containers
# Runs exact and pose_aware planners in parallel containers
#
# Usage: ./orchestrate_trials.sh [num_trials]
# Example: ./orchestrate_trials.sh 10

set -e

NUM_TRIALS="${1:-10}"
IMAGE_NAME="${AQUATIC_IMAGE:-aquatic-sim}"
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/trial_results}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "  Info Gain Trial Orchestrator"
echo "=============================================="
echo "  Num Trials:    $NUM_TRIALS (per field, 5 fields total)"
echo "  Docker Image:  $IMAGE_NAME"
echo "  Results Dir:   $RESULTS_DIR"
echo "  Scripts Dir:   $SCRIPT_DIR"
echo "=============================================="

# Create results directory structure
mkdir -p "$RESULTS_DIR/exact"
mkdir -p "$RESULTS_DIR/pose_aware"
mkdir -p "$RESULTS_DIR/logs"
mkdir -p "$RESULTS_DIR/monitor"

# Check if image exists
if ! docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
    echo "[ERROR] Docker image '$IMAGE_NAME' not found."
    echo "Build it first: cd infra/docker && docker build -t $IMAGE_NAME ."
    exit 1
fi

# Stop any existing containers
echo "[CLEANUP] Stopping any existing trial containers..."
docker stop exact_trials pose_aware_trials 2>/dev/null || true
docker rm exact_trials pose_aware_trials 2>/dev/null || true

# Initialize combined status file for monitor
MONITOR_FILE="$RESULTS_DIR/monitor/status.json"
cat > "$MONITOR_FILE" << EOF
{
    "start_time": "$(date -Iseconds)",
    "num_trials": $NUM_TRIALS,
    "exact": {"state": "starting", "current_trial": 0, "current_field": "", "message": "Initializing..."},
    "pose_aware": {"state": "starting", "current_trial": 0, "current_field": "", "message": "Initializing..."}
}
EOF

# Function to run a container
run_container() {
    local PLANNER=$1
    local CONTAINER_NAME="${PLANNER}_trials"

    echo "[START] Starting $PLANNER container..."

    # Set VNC ports based on planner type (use 59xx to avoid conflicts)
    if [[ "$PLANNER" == "exact" ]]; then
        VNC_PORT=5910
        NOVNC_PORT=6090
    else
        VNC_PORT=5911
        NOVNC_PORT=6091
    fi

    docker run -d \
        --name $CONTAINER_NAME \
        --gpus all \
        --entrypoint /bin/bash \
        -e PLANNER_TYPE=$PLANNER \
        -e NUM_TRIALS=$NUM_TRIALS \
        -e WORKSPACE_DIR="${CONTAINER_WS:-/home/simuser/aquatic-mapping}" \
        -e PX4_DIR="${CONTAINER_PX4:-/home/simuser/PX4-Autopilot}" \
        -p $VNC_PORT:$VNC_PORT \
        -p $NOVNC_PORT:$NOVNC_PORT \
        -v "$RESULTS_DIR/$PLANNER:${CONTAINER_WS:-/home/simuser/aquatic-mapping}/src/info_gain/data/trials/$PLANNER" \
        -v "$SCRIPT_DIR/run_all_trials.sh:/tmp/run_all_trials.sh:ro" \
        $IMAGE_NAME \
        -c "cd ${CONTAINER_WS:-/home/simuser/aquatic-mapping} && \
            source /opt/ros/jazzy/setup.bash && \
            source install/setup.bash && \
            /bin/bash /tmp/run_all_trials.sh $PLANNER $NUM_TRIALS"

    echo "[STARTED] $PLANNER container: $CONTAINER_NAME"
}

# Start both containers
run_container "exact"
run_container "pose_aware"

echo ""
echo "[RUNNING] Both containers started."
echo ""
echo "  Monitor dashboard:"
echo "    ./monitor_dashboard.py $RESULTS_DIR"
echo ""
echo "  View RViz2/matplotlib (noVNC):"
echo "    Exact:      http://localhost:6090/vnc.html"
echo "    Pose-aware: http://localhost:6091/vnc.html"
echo ""
echo "  View container logs:"
echo "    docker logs -f exact_trials"
echo "    docker logs -f pose_aware_trials"
echo ""

# Background process to aggregate status from both containers
(
    while true; do
        # Check container states
        EXACT_RUNNING=$(docker ps --format '{{.Names}}' | grep -q "^exact_trials$" && echo "true" || echo "false")
        POSE_RUNNING=$(docker ps --format '{{.Names}}' | grep -q "^pose_aware_trials$" && echo "true" || echo "false")

        # Read status files from mounted volumes
        EXACT_STATUS="{}"
        POSE_STATUS="{}"

        if [[ -f "$RESULTS_DIR/exact/status.json" ]]; then
            EXACT_STATUS=$(cat "$RESULTS_DIR/exact/status.json")
        fi
        if [[ -f "$RESULTS_DIR/pose_aware/status.json" ]]; then
            POSE_STATUS=$(cat "$RESULTS_DIR/pose_aware/status.json")
        fi

        # Count completed trials
        EXACT_COMPLETED=$(find "$RESULTS_DIR/exact" -name "summary.json" 2>/dev/null | wc -l)
        POSE_COMPLETED=$(find "$RESULTS_DIR/pose_aware" -name "summary.json" 2>/dev/null | wc -l)

        # Write combined status
        cat > "$MONITOR_FILE" << EOF
{
    "start_time": "$(date -Iseconds)",
    "num_trials": $NUM_TRIALS,
    "total_expected": $((NUM_TRIALS * 5 * 2)),
    "exact": {
        "container_running": $EXACT_RUNNING,
        "completed_count": $EXACT_COMPLETED,
        "status": $EXACT_STATUS
    },
    "pose_aware": {
        "container_running": $POSE_RUNNING,
        "completed_count": $POSE_COMPLETED,
        "status": $POSE_STATUS
    },
    "updated": "$(date -Iseconds)"
}
EOF

        # Exit if both containers stopped
        if [[ "$EXACT_RUNNING" == "false" && "$POSE_RUNNING" == "false" ]]; then
            echo "[DONE] Both containers have finished"
            break
        fi

        sleep 5
    done
) &
MONITOR_PID=$!

echo "[MONITOR] Status aggregator running (PID: $MONITOR_PID)"
echo "  Status file: $MONITOR_FILE"
echo ""
echo "  Press Ctrl+C to stop monitoring (containers will continue running)"
echo ""

# Wait for user interrupt or completion
wait $MONITOR_PID 2>/dev/null || true

# Copy container logs
echo "[LOGS] Saving container logs..."
docker logs exact_trials > "$RESULTS_DIR/logs/exact_container.log" 2>&1 || true
docker logs pose_aware_trials > "$RESULTS_DIR/logs/pose_aware_container.log" 2>&1 || true

echo ""
echo "=============================================="
echo "  ORCHESTRATOR COMPLETE"
echo "=============================================="
echo "  Results: $RESULTS_DIR"
echo "=============================================="
