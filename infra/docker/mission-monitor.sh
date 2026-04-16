#!/bin/bash
# Monitor mission completion and stop container
# Checks if lawnmower mission has completed

WORKSPACE_DIR="${WORKSPACE_DIR:-/home/simuser/aquatic-mapping}"
TRIAL_NUMBER="${1:-1}"

echo "[MONITOR] Starting mission completion monitor for trial $TRIAL_NUMBER..."
echo "[MONITOR] Waiting 30s for mission to start..."
sleep 30

# Monitor for mission completion
# The mission is complete when the rover finishes the lawnmower pattern
# We'll check if the lawnmower node is still running

while true; do
    sleep 5

    # Check if lawnmower process is still running
    if ! pgrep -f "lawnmower" > /dev/null; then
        echo "[MONITOR] Lawnmower mission process not found, checking if it completed..."
        sleep 10

        # Double check it's really done
        if ! pgrep -f "lawnmower" > /dev/null; then
            echo "[MONITOR] Mission completed! Waiting 10s for data to flush..."
            sleep 10

            echo "[MONITOR] Stopping all processes..."
            # Kill all ROS2 processes
            pkill -f ros2
            pkill -f px4
            pkill -f gz
            pkill -f MicroXRCEAgent

            echo "[MONITOR] Mission complete. Container will exit."
            exit 0
        fi
    fi
done
