#!/usr/bin/env python3


from launch import LaunchDescription
from launch.actions import LogInfo, TimerAction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    control_pkg_dir = get_package_share_directory('control')
    info_gain_pkg_dir = get_package_share_directory('info_gain')

    # --- Load Rover URDF ---
    urdf_file = os.path.join(control_pkg_dir, 'urdf', 'r1_rover.urdf')
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # --- RViz config ---
    rviz_config = os.path.join(info_gain_pkg_dir, 'config', 'mi_field.rviz')

    planner_arg = DeclareLaunchArgument(
        'planner',
        default_value=TextSubstitution(text='1'),
        description='MI planner version to run (e.g., 1, 2, 3)'
    )

    return LaunchDescription([
        planner_arg,
        # World frame publisher (world -> map)
        Node(
            package='control',
            executable='world_frame_publisher.py',
            name='world_frame_publisher',
            output='screen',
        ),

        # Static TF world -> odom
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='world_to_odom',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'odom'],
            output='screen',
        ),

        # Rover state publisher (delayed slightly so TF is ready)
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

        # PX4 odometry to TF broadcaster (for RViz yaw visualization)
        Node(
            package='info_gain',
            executable='px4_odom_to_tf.py',
            name='px4_odom_tf',
            output='log',
            arguments=['--ros-args', '--log-level', 'warn']
        ),

        # Radial field (ground truth)
        Node(
            package='control',
            executable='radial_field.py',
            name='radial_field_generator',
            output='screen',
            parameters=[{'noise_std': 0.2}],
        ),

        # MI planner (selectable version)
        Node(
            package='info_gain',
            executable=[TextSubstitution(text='mi_planner_'), LaunchConfiguration('planner'), TextSubstitution(text='.py')],
            name=[TextSubstitution(text='mi_planner_'), LaunchConfiguration('planner')],
            output='screen',
            parameters=[
                {'width': 25.0},
                {'height': 25.0},
                {'resolution': 1.0},
                {'ell': 5.0},
                {'sigma_f': 4.0},
                {'sigma_n': 0.2},
                {'plan_period': 0.5},
                {'step_size': 0.75},
                {'candidate_subsample': 600},
                {'frame_id': 'world'},
            ],
        ),

        # PX4 relay to follow MI waypoints
        Node(
            package='info_gain',
            executable='mi_to_px4_relay.py',
            name='mi_to_px4_relay',
            output='screen',
        ),

        # RViz2 visualization
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
        ),
    ])
