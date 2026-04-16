#!/bin/bash
# Run multiple trials in parallel
# All 5 field types run automatically for each trial
#
# Usage: ./run-batch.sh <start_trial> <end_trial> [max_concurrent]
#
# Examples:
#   ./run-batch.sh 1 5           # Run trials 1-5, max 5 concurrent
#   ./run-batch.sh 1 25 8        # Run trials 1-25, max 8 concurrent
#   ./run-batch.sh 26 50 3       # Run trials 26-50, max 3 concurrent

set -e

START_TRIAL="${1}"
END_TRIAL="${2}"
MAX_CONCURRENT="${3:-5}"

if [ -z "$START_TRIAL" ] || [ -z "$END_TRIAL" ]; then
    echo "Error: Start and end trial numbers required"
    echo "Usage: $0 <start_trial> <end_trial> [max_concurrent]"
    echo ""
    echo "Examples:"
    echo "  $0 1 5           # Run trials 1-5, max 5 concurrent"
    echo "  $0 1 25 8        # Run trials 1-25, max 8 concurrent"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "  Batch Trial Execution"
echo "=============================================="
echo "  Trials:         $START_TRIAL to $END_TRIAL"
echo "  Fields:         All 5 (per trial)"
echo "  Max Concurrent: $MAX_CONCURRENT"
echo "=============================================="
echo ""

# Function to count running trial containers
count_running() {
    docker ps --filter "name=aquatic-trial-" --format '{{.Names}}' | wc -l
}

# Function to wait for slot availability
wait_for_slot() {
    while [ $(count_running) -ge $MAX_CONCURRENT ]; do
        echo "  [$(date +%H:%M:%S)] Waiting for slot... ($(count_running)/$MAX_CONCURRENT running)"
        sleep 5
    done
}

# Launch trials
for TRIAL_ID in $(seq $START_TRIAL $END_TRIAL); do
    wait_for_slot

    echo "[$(date +%H:%M:%S)] Starting trial $TRIAL_ID..."
    "$SCRIPT_DIR/run-trial.sh" "$TRIAL_ID" > /dev/null 2>&1

    # Brief pause to avoid overwhelming Docker
    sleep 2
done

echo ""
echo "=============================================="
echo "  All trials queued!"
echo "=============================================="
echo ""
echo "Monitor running containers:"
echo "  watch -n 2 'docker ps --filter \"name=aquatic-trial-\"'"
echo ""
echo "View all trial logs:"
echo "  docker ps --filter \"name=aquatic-trial-\" --format '{{.Names}}' | xargs -I {} echo \"docker logs -f {}\""
echo ""
echo "Stop all trials:"
echo "  docker ps --filter \"name=aquatic-trial-\" --format '{{.Names}}' | xargs docker stop"
echo ""
echo "=============================================="
