#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import LogInfo
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    
    # Prefer user's source config path; fall back to installed package path
    pkg_dir = get_package_share_directory('control')
    src_config = '/home/dreams-lab-u24/workspaces/aquatic-mapping/src/control/config/five_fields.rviz'
    installed_config = os.path.join(pkg_dir, 'config', 'five_fields.rviz')
    rviz_config = src_config if os.path.exists(src_config) else installed_config

    return LaunchDescription([
        
        # World frame publisher (establishes world frame in TF tree)
        Node(
            package='control',
            executable='world_frame_publisher.py',
            name='world_frame_publisher',
            output='screen',
        ),
        
        # Radial field (isotropic Gaussian)
        Node(
            package='control',
            executable='radial_field.py',
            name='radial_field_generator',
            output='screen',
        ),
        
        # X-Compress field (compressed along X-axis)
        Node(
            package='control',
            executable='x_compress_field.py',
            name='x_compress_field_generator',
            output='screen',
        ),
        
        # X-Compress-Tilt field (compressed X, rotated 45°)
        Node(
            package='control',
            executable='x_compress_tilt_field.py',
            name='x_compress_tilt_field_generator',
            output='screen',
        ),
        
        # Y-Compress field (compressed along Y-axis)
        Node(
            package='control',
            executable='y_compress_field.py',
            name='y_compress_field_generator',
            output='screen',
        ),
        
        # Y-Compress-Tilt field (compressed Y, rotated 45°)
        Node(
            package='control',
            executable='y_compress_tilt_field.py',
            name='y_compress_tilt_field_generator',
            output='screen',
        ),

        # RViz2 with provided config
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
        ),
    ])

