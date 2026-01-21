#!/bin/bash
# Clean up trial data and reconstruction results
# Usage: ./cleanup-trials.sh TRIAL_ID [TRIAL_ID...]
# Example: ./cleanup-trials.sh 7 9

if [ $# -eq 0 ]; then
    echo "Usage: $0 TRIAL_ID [TRIAL_ID...]"
    echo "Example: $0 7 9"
    exit 1
fi

MISSIONS_DIR="$HOME/workspaces/aquatic-mapping/src/sampling/data/missions"
RESULTS_DIR="$HOME/workspaces/aquatic-mapping/reconstruction/results"

for TRIAL_ID in "$@"; do
    echo "Deleting trial $TRIAL_ID..."

    # Delete trial data (may need sudo for Docker-created files)
    TRIAL_DATA="$MISSIONS_DIR/trial_$TRIAL_ID"
    if [ -d "$TRIAL_DATA" ]; then
        if rm -rf "$TRIAL_DATA" 2>/dev/null; then
            echo "  ✓ Deleted trial data: $TRIAL_DATA"
        else
            echo "  ✗ Permission denied. Trying with sudo..."
            sudo rm -rf "$TRIAL_DATA"
            if [ $? -eq 0 ]; then
                echo "  ✓ Deleted trial data with sudo: $TRIAL_DATA"
            else
                echo "  ✗ Failed to delete trial data"
            fi
        fi
    else
        echo "  - No trial data found"
    fi

    # Delete reconstruction results
    TRIAL_RESULTS="$RESULTS_DIR/trial_$TRIAL_ID"
    if [ -d "$TRIAL_RESULTS" ]; then
        rm -rf "$TRIAL_RESULTS"
        echo "  ✓ Deleted reconstruction results: $TRIAL_RESULTS"
    else
        echo "  - No reconstruction results found"
    fi

    echo ""
done

echo "Cleanup complete!"
