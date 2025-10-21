#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import LogInfo, TimerAction
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('control')
    urdf_file = os.path.join(pkg_dir, 'urdf', 'r1_rover.urdf')
    
    with open(urdf_file, 'r') as f:
        robot_description = f.read()
    
    # RViz config - prioritize source directory
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(pkg_dir))))
    rviz_src_config = os.path.join(workspace_root, 'src', 'control', 'config', 'five_fields.rviz')
    rviz_installed_config = os.path.join(pkg_dir, 'config', 'five_fields.rviz')
    
    if os.path.exists(rviz_src_config):
        rviz_config = rviz_src_config
    else:
        rviz_config = rviz_installed_config
    
    return LaunchDescription([
     
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='world_to_odom',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'odom'],
            output='screen',
        ),
        
        # Rover TF and state
        Node(
            package='control',
            executable='rover_monitor.py',
            name='rover_monitor',
            output='screen',
            emulate_tty=True
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
        
        # RViz with five_fields config
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen'
        ),
    ])