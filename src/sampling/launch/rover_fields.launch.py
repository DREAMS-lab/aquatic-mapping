#!/usr/bin/env python3
"""
Rover with field generators launch file.
Launches:
  - Everything from rover.launch.py
  - All 5 field generators (radial, x_compress, y_compress, x_compress_tilt, y_compress_tilt)

Usage:
  ros2 launch sampling rover_fields.launch.py
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
        # ========== ROVER INFRASTRUCTURE ==========

        # Static TF: world -> odom
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='world_to_odom',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'odom'],
            output='screen',
        ),

        # Rover monitor (with proper quaternion conversion)
        Node(
            package='sampling',
            executable='rover_monitor.py',
            name='rover_monitor',
            output='screen',
            emulate_tty=True
        ),

        # Robot state publisher (delayed)
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

        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen'
        ),

        # ========== FIELD GENERATORS ==========

        Node(
            package='sampling',
            executable='radial_field.py',
            name='radial_field',
            output='screen'
        ),

        Node(
            package='sampling',
            executable='x_compress_field.py',
            name='x_compress_field',
            output='screen'
        ),

        Node(
            package='sampling',
            executable='y_compress_field.py',
            name='y_compress_field',
            output='screen'
        ),

        Node(
            package='sampling',
            executable='x_compress_tilt_field.py',
            name='x_compress_tilt_field',
            output='screen'
        ),

        Node(
            package='sampling',
            executable='y_compress_tilt_field.py',
            name='y_compress_tilt_field',
            output='screen'
        ),
    ])
