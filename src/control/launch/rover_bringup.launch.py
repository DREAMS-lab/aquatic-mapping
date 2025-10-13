#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('control')
    
    # URDF file path
    urdf_file = os.path.join(pkg_dir, 'urdf', 'r1_rover.urdf')
    
    # RViz config path
    rviz_config = os.path.join(pkg_dir, 'config', 'rover_viz.rviz')
    
    # Read URDF
    with open(urdf_file, 'r') as file:
        robot_description = file.read()
    
    # Static transform: world -> odom
    world_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'odom'],
        output='screen'
    )
    
    # Static transform: odom -> field (so temperature field aligns with rover)
    odom_to_field = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='odom_to_field',
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'field'],
        output='screen'
    )
    
    # Rover Monitor - publishes odom -> base_link TF
    rover_monitor = Node(
        package='control',
        executable='rover_monitor.py',
        name='rover_monitor',
        output='screen',
        emulate_tty=True
    )
    
    # Robot State Publisher - publishes base_link -> all other links
    robot_state_publisher = TimerAction(
        period=1.0,  # Wait 1 second for rover_monitor to start
        actions=[
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                name='robot_state_publisher',
                parameters=[{
                    'robot_description': robot_description,
                    'frame_prefix': '',
                    'publish_frequency': 20.0,
                }],
                output='screen'
            )
        ]
    )
    
    # Temperature Field Node
    field_node = Node(
        package='control',
        executable='field.py',
        name='field',
        output='screen',
        emulate_tty=True
    )
    
    # RViz2 with config
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )
    
    return LaunchDescription([
        world_to_odom,      # world -> odom
        odom_to_field,      # odom -> field 
        rover_monitor,      # Publishes: odom -> base_link
        robot_state_publisher,  # Publishes: base_link -> wheels
        field_node,
        rviz,
    ])