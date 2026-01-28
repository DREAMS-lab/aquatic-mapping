#!/bin/bash
# Stop all info gain simulation processes

echo "Stopping info gain simulation..."

pkill -f "px4" && echo "✓ Killed PX4" || true
pkill -f "MicroXRCEAgent" && echo "✓ Killed DDS Agent" || true
pkill -f "gz" && echo "✓ Killed Gazebo" || true
pkill -f "ruby.*gz" && echo "✓ Killed Gazebo Ruby" || true
pkill -f "rviz2" && echo "✓ Killed RViz" || true
pkill -f "exact_planner" && echo "✓ Killed Exact Planner" || true
pkill -f "pose_aware_planner" && echo "✓ Killed Pose-Aware Planner" || true
pkill -f "_field" && echo "✓ Killed Field Generator" || true

sleep 1
echo "Done"
