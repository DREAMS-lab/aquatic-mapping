#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction, LogInfo
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    # Declare arguments
    trial_number_arg = DeclareLaunchArgument(
        'trial_number',
        default_value='1',
        description='Trial number (1-10)'
    )
    
    trial_number = LaunchConfiguration('trial_number')
    
    return LaunchDescription([
        trial_number_arg,
        
        LogInfo(msg='========================================================='),
        LogInfo(msg=' Starting Recording Mission'),
        LogInfo(msg='========================================================='),
        
        # Start all 5 field generators immediately
        Node(
            package='control',
            executable='radial_field.py',
            name='radial_field',
            output='screen'
        ),
        Node(
            package='control',
            executable='x_compress_field.py',
            name='x_compress_field',
            output='screen'
        ),
        Node(
            package='control',
            executable='x_compress_tilt_field.py',
            name='x_compress_tilt_field',
            output='screen'
        ),
        Node(
            package='control',
            executable='y_compress_field.py',
            name='y_compress_field',
            output='screen'
        ),
        Node(
            package='control',
            executable='y_compress_tilt_field.py',
            name='y_compress_tilt_field',
            output='screen'
        ),
        
        # Wait 3 seconds for rover to reach (0,0), then start filters
        TimerAction(
            period=3.0,
            actions=[
                LogInfo(msg='Starting data filters (1m spacing)...'),
                Node(
                    package='control',
                    executable='field_filter.py',
                    name='radial_filter',
                    arguments=['radial'],
                    output='screen'
                ),
                Node(
                    package='control',
                    executable='field_filter.py',
                    name='x_compress_filter',
                    arguments=['x_compress'],
                    output='screen'
                ),
                Node(
                    package='control',
                    executable='field_filter.py',
                    name='x_compress_tilt_filter',
                    arguments=['x_compress_tilt'],
                    output='screen'
                ),
                Node(
                    package='control',
                    executable='field_filter.py',
                    name='y_compress_filter',
                    arguments=['y_compress'],
                    output='screen'
                ),
                Node(
                    package='control',
                    executable='field_filter.py',
                    name='y_compress_tilt_filter',
                    arguments=['y_compress_tilt'],
                    output='screen'
                ),
            ]
        ),
        
        # Wait 5 seconds, then start recorders and lawnmower
        TimerAction(
            period=5.0,
            actions=[
                LogInfo(msg='Starting rosbag recorders...'),
                Node(
                    package='control',
                    executable='record_field_data.py',
                    name='radial_recorder',
                    arguments=['radial', trial_number],
                    output='screen'
                ),
                Node(
                    package='control',
                    executable='record_field_data.py',
                    name='x_compress_recorder',
                    arguments=['x_compress', trial_number],
                    output='screen'
                ),
                Node(
                    package='control',
                    executable='record_field_data.py',
                    name='x_compress_tilt_recorder',
                    arguments=['x_compress_tilt', trial_number],
                    output='screen'
                ),
                Node(
                    package='control',
                    executable='record_field_data.py',
                    name='y_compress_recorder',
                    arguments=['y_compress', trial_number],
                    output='screen'
                ),
                Node(
                    package='control',
                    executable='record_field_data.py',
                    name='y_compress_tilt_recorder',
                    arguments=['y_compress_tilt', trial_number],
                    output='screen'
                ),
                LogInfo(msg='Starting lawnmower mission...'),
                Node(
                    package='control',
                    executable='lawnmower.py',
                    name='lawnmower_mission',
                    output='screen'
                )
            ]
        ),
    ])