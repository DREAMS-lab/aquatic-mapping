#!/bin/bash
# Test a single container simulation
#
# Usage: ./test-single.sh [mode]
#   modes: shell, rover, headless-test, mission (default: shell)

set -e

MODE="${1:-shell}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$SCRIPT_DIR/../docker"

echo "=============================================="
echo "  Testing Aquatic Simulation Container"
echo "=============================================="
echo "  Mode: $MODE"
echo "=============================================="

# Check if image exists
if ! docker image inspect aquatic-sim:latest >/dev/null 2>&1; then
    echo "ERROR: Image 'aquatic-sim:latest' not found."
    echo "Run: ./build-image.sh first"
    exit 1
fi

# Clean up any existing test container
docker rm -f aquatic-sim-test 2>/dev/null || true

case "$MODE" in
    shell)
        echo "Starting interactive shell..."
        docker run -it --rm \
            --name aquatic-sim-test \
            -e DISPLAY=:99 \
            -p 6080:6080 \
            aquatic-sim:latest shell
        ;;

    rover)
        echo "Starting rover stack (detached)..."
        docker run -d \
            --name aquatic-sim-test \
            -e TRIAL_ID=1 \
            -e FIELD_TYPE=radial \
            -e ROS_DOMAIN_ID=99 \
            -e HEADLESS=1 \
            -p 6080:6080 \
            aquatic-sim:latest rover

        echo ""
        echo "Container started! Access:"
        echo "  - noVNC: http://localhost:6080/vnc.html"
        echo "  - Logs:  docker logs -f aquatic-sim-test"
        echo "  - Shell: docker exec -it aquatic-sim-test bash"
        echo "  - Stop:  docker stop aquatic-sim-test"
        ;;

    headless-test)
        echo "Running headless test (30 seconds)..."
        docker run --rm \
            --name aquatic-sim-test \
            -e HEADLESS=1 \
            aquatic-sim:latest headless-test
        ;;

    mission)
        echo "Starting full mission..."
        docker run -d \
            --name aquatic-sim-test \
            -e TRIAL_ID=1 \
            -e FIELD_TYPE=radial \
            -e ROS_DOMAIN_ID=99 \
            -e HEADLESS=1 \
            -p 6080:6080 \
            -v "$SCRIPT_DIR/../../src/sampling/data/missions:/home/simuser/aquatic-mapping/src/sampling/data/missions" \
            aquatic-sim:latest mission

        echo ""
        echo "Mission started! Access:"
        echo "  - noVNC: http://localhost:6080/vnc.html"
        echo "  - Logs:  docker logs -f aquatic-sim-test"
        echo "  - Stop:  docker stop aquatic-sim-test"
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Valid modes: shell, rover, headless-test, mission"
        exit 1
        ;;
esac
