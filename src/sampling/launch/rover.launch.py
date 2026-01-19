#!/usr/bin/env python3
"""
Rover-only launch file.
Launches:
  - Static TF (world -> odom)
  - Rover monitor (TF: odom -> base_link, with proper quaternion conversion)
  - Robot state publisher (URDF)
  - RViz2

Usage:
  ros2 launch sampling rover.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('sampling')

    # Load URDF
    urdf_file = os.path.join(pkg_dir, 'urdf', 'r1_rover.urdf')
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # RViz config
    rviz_config = os.path.join(pkg_dir, 'config', 'rviz', 'rover.rviz')

    return LaunchDescription([
        # Static TF: world -> odom (identity transform)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='world_to_odom',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'odom'],
            output='screen',
        ),

        # Rover monitor: subscribes to PX4 odometry, publishes odom->base_link TF
        Node(
            package='sampling',
            executable='rover_monitor.py',
            name='rover_monitor',
            output='screen',
            emulate_tty=True
        ),

        # Robot state publisher (delayed 1s to ensure TF tree is ready)
        TimerAction(
            period=1.0,
            actions=[
                Node(
                    package='robot_state_publisher',
                    executable='robot_state_publisher',
                    name='robot_state_publisher',
                    parameters=[{
                        'robot_description': robot_description,
                        'frame_prefix': '',
                        'publish_frequency': 20.0
                    }],
                    output='screen',
                )
            ],
        ),

        # RViz2 visualization
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen'
        ),
    ])
