#!/bin/bash
# stop all rover simulation processes

echo "stopping rover simulation..."

# kill all processes
pkill -f "px4" && echo "✓ killed px4" || true
pkill -f "micro-xrce-dds-agent" && echo "✓ killed dds agent" || true
pkill -f "gz" && echo "✓ killed gazebo" || true
pkill -f "ruby.*gz" && echo "✓ killed gz ruby processes" || true
pkill -f "sampling.*launch" && echo "✓ killed ros2 launch" || true

sleep 1
echo "done"
