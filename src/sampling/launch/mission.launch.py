#!/usr/bin/env python3
"""
Full mission launch file.
Launches:
  - Everything from rover_fields.launch.py
  - Lawnmower mission (delayed 5s)
  - Data recorders for all 5 fields (delayed 5s)

Usage:
  ros2 launch sampling mission.launch.py
  ros2 launch sampling mission.launch.py trial_number:=2

Arguments:
  trial_number: Trial number for data organization (default: 1)
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction, LogInfo, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
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

    # Launch argument: trial number
    trial_number_arg = DeclareLaunchArgument(
        'trial_number',
        default_value='1',
        description='Trial number for data organization (1-10)'
    )
    trial_number = LaunchConfiguration('trial_number')

    return LaunchDescription([
        # Declare arguments
        trial_number_arg,

        # ========== ROVER INFRASTRUCTURE ==========

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='world_to_odom',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'odom'],
            output='screen',
        ),

        # Rover monitor (quiet mode for mission)
        Node(
            package='sampling',
            executable='rover_monitor.py',
            name='rover_monitor',
            output='log',
            emulate_tty=True,
            arguments=['--ros-args', '--log-level', 'warn']
        ),

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

        # ========== MISSION + DATA RECORDING (delayed 5s) ==========

        TimerAction(
            period=5.0,
            actions=[
                LogInfo(msg='='*60),
                LogInfo(msg='Starting mission and data recorders...'),
                LogInfo(msg='='*60),

                # Data recorders for all 5 fields
                Node(
                    package='sampling',
                    executable='record_field_data.py',
                    name='radial_recorder',
                    arguments=['radial', trial_number],
                    output='screen'
                ),
                Node(
                    package='sampling',
                    executable='record_field_data.py',
                    name='x_compress_recorder',
                    arguments=['x_compress', trial_number],
                    output='screen'
                ),
                Node(
                    package='sampling',
                    executable='record_field_data.py',
                    name='y_compress_recorder',
                    arguments=['y_compress', trial_number],
                    output='screen'
                ),
                Node(
                    package='sampling',
                    executable='record_field_data.py',
                    name='x_compress_tilt_recorder',
                    arguments=['x_compress_tilt', trial_number],
                    output='screen'
                ),
                Node(
                    package='sampling',
                    executable='record_field_data.py',
                    name='y_compress_tilt_recorder',
                    arguments=['y_compress_tilt', trial_number],
                    output='screen'
                ),

                # Lawnmower mission
                Node(
                    package='sampling',
                    executable='lawnmower.py',
                    name='lawnmower_mission',
                    output='screen'
                )
            ]
        ),
    ])
