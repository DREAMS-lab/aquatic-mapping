#!/bin/bash
# Run a single trial simulation in a container
# Runs all 5 field types automatically
#
# Usage: ./run-trial.sh <trial_number>
#
# Examples:
#   ./run-trial.sh 1     # Trial 1 (all fields)
#   ./run-trial.sh 25    # Trial 25 (all fields)

set -e

TRIAL_ID="${1}"

if [ -z "$TRIAL_ID" ]; then
    echo "Error: Trial number required"
    echo "Usage: $0 <trial_number>"
    echo ""
    echo "Examples:"
    echo "  $0 1     # Run trial 1 (all 5 fields)"
    echo "  $0 25    # Run trial 25 (all 5 fields)"
    exit 1
fi

# Directories
HOST_DATA_DIR="$HOME/workspaces/aquatic-mapping/src/sampling/data/missions"
CONTAINER_NAME="aquatic-trial-${TRIAL_ID}"
DOMAIN_ID=$((TRIAL_ID % 100))  # Keep ROS domain ID reasonable
NOVNC_PORT=$((6080 + TRIAL_ID))

echo "=============================================="
echo "  Starting Aquatic Mapping Trial"
echo "=============================================="
echo "  Trial ID:       $TRIAL_ID"
echo "  Fields:         All 5 (radial, x_compress, y_compress, x_compress_tilt, y_compress_tilt)"
echo "  ROS_DOMAIN_ID:  $DOMAIN_ID"
echo "  Container:      $CONTAINER_NAME"
echo "  Data saved to:  $HOST_DATA_DIR/trial_${TRIAL_ID}/"
echo "  noVNC:          http://localhost:$NOVNC_PORT/vnc.html"
echo "=============================================="

# Create data directory with proper permissions
mkdir -p "$HOST_DATA_DIR/trial_${TRIAL_ID}"
chmod -R 777 "$HOST_DATA_DIR"

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo ""
    echo "Warning: Container '$CONTAINER_NAME' already exists"
    read -p "Stop and remove it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
    else
        echo "Aborted."
        exit 1
    fi
fi

# Run the container
echo ""
echo "Starting container..."
docker run -d \
    --name "$CONTAINER_NAME" \
    -e TRIAL_ID="$TRIAL_ID" \
    -e ROS_DOMAIN_ID="$DOMAIN_ID" \
    -e HEADLESS=1 \
    -v "$HOST_DATA_DIR:/home/simuser/aquatic-mapping/src/sampling/data/missions" \
    -p "$NOVNC_PORT:6080" \
    aquatic-sim:latest mission

echo ""
echo "=============================================="
echo "  Trial Started Successfully!"
echo "=============================================="
echo ""
echo "  Access GUI:  http://localhost:$NOVNC_PORT/vnc.html"
echo "  View logs:   docker logs -f $CONTAINER_NAME"
echo "  Stop trial:  docker stop $CONTAINER_NAME"
echo ""
echo "  Data output: $HOST_DATA_DIR/trial_${TRIAL_ID}/"
echo "    trial_${TRIAL_ID}/radial/"
echo "    trial_${TRIAL_ID}/x_compress/"
echo "    trial_${TRIAL_ID}/y_compress/"
echo "    trial_${TRIAL_ID}/x_compress_tilt/"
echo "    trial_${TRIAL_ID}/y_compress_tilt/"
echo ""
echo "  Monitor all fields:"
echo "    watch -n 5 'ls -lh $HOST_DATA_DIR/trial_${TRIAL_ID}/'"
echo ""
echo "=============================================="
