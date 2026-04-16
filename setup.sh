#!/bin/bash
# One-command workspace setup for aquatic-mapping
# Usage: ./setup.sh [--gpu]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Aquatic Mapping Workspace Setup ==="

# 1. Git submodules
echo "[1/4] Initializing submodules (px4_msgs)..."
git submodule update --init --recursive

# 2. Python venv (at workspace root, outside src/ so colcon doesn't pick it up)
echo "[2/4] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

if [[ "$1" == "--gpu" ]]; then
    echo "       Installing with CUDA 12.4 GPU support..."
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
else
    echo "       Installing CPU-only packages..."
    pip install -r requirements.txt
fi

# 3. Data directories
echo "[3/4] Creating data directories..."
mkdir -p data/{trials,reconstruction,statistics}

# 4. Build ROS2 packages
echo "[4/4] Building ROS2 packages..."
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
    colcon build --symlink-install \
        --packages-select px4_msgs sampling info_gain nonstationary_planning
    echo ""
    echo "=== Setup complete ==="
    echo ""
    echo "To use:"
    echo "  source venv/bin/activate"
    echo "  source install/setup.bash"
    echo "  ros2 launch info_gain exact.launch.py field_type:=radial trial:=1"
else
    echo ""
    echo "=== Partial setup complete (ROS2 Jazzy not found) ==="
    echo ""
    echo "Python venv is ready. Install ROS2 Jazzy to build and run planners."
    echo "See: https://docs.ros.org/en/jazzy/Installation.html"
fi
