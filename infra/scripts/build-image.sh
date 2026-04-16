#!/bin/bash
# Build the aquatic-sim Docker image
#
# Usage: ./build-image.sh [--no-cache]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$SCRIPT_DIR/../docker"

echo "=============================================="
echo "  Building Aquatic Simulation Docker Image"
echo "=============================================="

cd "$DOCKER_DIR"

# Check for --no-cache flag
CACHE_FLAG=""
if [ "$1" = "--no-cache" ]; then
    CACHE_FLAG="--no-cache"
    echo "Building with --no-cache"
fi

# Build the image
docker build $CACHE_FLAG -t aquatic-sim:latest .

echo ""
echo "=============================================="
echo "  Build Complete!"
echo "=============================================="
echo ""
echo "Image: aquatic-sim:latest"
echo ""
echo "To run a single simulation:"
echo "  docker run -d --name sim1 -e TRIAL_ID=1 -e FIELD_TYPE=radial -p 6081:6080 aquatic-sim:latest"
echo ""
echo "To run multiple simulations with docker-compose:"
echo "  cd $DOCKER_DIR && docker compose up sim1 sim2 sim3"
echo ""
