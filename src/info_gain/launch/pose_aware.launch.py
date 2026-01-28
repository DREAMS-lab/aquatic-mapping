#!/usr/bin/env python3
"""
Pose-Aware Planner Launch File

Launches everything needed for pose-aware planning experiment (position uncertainty):
  - Rover infrastructure (TF, rover_monitor, robot_state_publisher)
  - Selected field generator ONLY
  - RViz with field-specific config
  - Pose-aware planner node (expected info under position noise)

Usage:
  ros2 launch info_gain pose_aware.launch.py field_type:=radial
  ros2 launch info_gain pose_aware.launch.py field_type:=radial trial:=1 position_std:=0.5

Available field types: radial, x_compress, y_compress, x_compress_tilt, y_compress_tilt
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package directories
    sampling_pkg_dir = get_package_share_directory('sampling')
    info_gain_pkg_dir = get_package_share_directory('info_gain')

    # Load URDF
    urdf_file = os.path.join(sampling_pkg_dir, 'urdf', 'r1_rover.urdf')
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # RViz config path will be constructed dynamically based on field_type
    rviz_config_base = os.path.join(info_gain_pkg_dir, 'config', 'rviz')

    # Declare launch arguments
    field_type_arg = DeclareLaunchArgument(
        'field_type',
        default_value='radial',
        description='Field type: radial, x_compress, y_compress, x_compress_tilt, y_compress_tilt'
    )

    trial_arg = DeclareLaunchArgument(
        'trial',
        default_value='-1',
        description='Trial number (-1 = auto-increment)'
    )

    horizon_arg = DeclareLaunchArgument(
        'horizon',
        default_value='2',
        description='Planning horizon H (1=greedy, 2=default)'
    )

    lambda_cost_arg = DeclareLaunchArgument(
        'lambda_cost',
        default_value='0.1',
        description='Trade-off: score = info_gain - lambda * travel_cost'
    )

    noise_var_arg = DeclareLaunchArgument(
        'noise_var',
        default_value='0.01',
        description='GP observation noise variance'
    )

    lengthscale_arg = DeclareLaunchArgument(
        'lengthscale',
        default_value='2.0',
        description='GP RBF kernel lengthscale (meters)'
    )

    candidate_resolution_arg = DeclareLaunchArgument(
        'candidate_resolution',
        default_value='1.0',
        description='Candidate grid spacing (meters)'
    )

    position_std_arg = DeclareLaunchArgument(
        'position_std',
        default_value='0.5',
        description='Position uncertainty std deviation (meters)'
    )

    n_mc_samples_arg = DeclareLaunchArgument(
        'n_mc_samples',
        default_value='30',
        description='Monte Carlo samples for expected info gain'
    )

    # Get launch configurations
    field_type = LaunchConfiguration('field_type')
    trial = LaunchConfiguration('trial')
    horizon = LaunchConfiguration('horizon')
    lambda_cost = LaunchConfiguration('lambda_cost')
    noise_var = LaunchConfiguration('noise_var')
    lengthscale = LaunchConfiguration('lengthscale')
    candidate_resolution = LaunchConfiguration('candidate_resolution')
    position_std = LaunchConfiguration('position_std')
    n_mc_samples = LaunchConfiguration('n_mc_samples')

    return LaunchDescription([
        # ========== LAUNCH ARGUMENTS ==========
        field_type_arg,
        trial_arg,
        horizon_arg,
        lambda_cost_arg,
        noise_var_arg,
        lengthscale_arg,
        candidate_resolution_arg,
        position_std_arg,
        n_mc_samples_arg,

        # ========== ROVER INFRASTRUCTURE ==========

        # Static TF: world -> odom
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='world_to_odom',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'odom'],
            output='screen',
        ),

        # Rover monitor (TF: odom -> base_link)
        Node(
            package='sampling',
            executable='rover_monitor.py',
            name='rover_monitor',
            output='screen',
            emulate_tty=True
        ),

        # Robot state publisher (delayed 1s)
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

        # ========== FIELD GENERATORS (only selected field) ==========

        Node(
            package='sampling',
            executable='radial_field.py',
            name='radial_field',
            output='screen',
            condition=IfCondition(PythonExpression(["'", field_type, "' == 'radial'"]))
        ),

        Node(
            package='sampling',
            executable='x_compress_field.py',
            name='x_compress_field',
            output='screen',
            condition=IfCondition(PythonExpression(["'", field_type, "' == 'x_compress'"]))
        ),

        Node(
            package='sampling',
            executable='y_compress_field.py',
            name='y_compress_field',
            output='screen',
            condition=IfCondition(PythonExpression(["'", field_type, "' == 'y_compress'"]))
        ),

        Node(
            package='sampling',
            executable='x_compress_tilt_field.py',
            name='x_compress_tilt_field',
            output='screen',
            condition=IfCondition(PythonExpression(["'", field_type, "' == 'x_compress_tilt'"]))
        ),

        Node(
            package='sampling',
            executable='y_compress_tilt_field.py',
            name='y_compress_tilt_field',
            output='screen',
            condition=IfCondition(PythonExpression(["'", field_type, "' == 'y_compress_tilt'"]))
        ),

        # ========== RVIZ (field-specific config) ==========

        TimerAction(
            period=2.0,
            actions=[
                # Radial
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    arguments=['-d', os.path.join(rviz_config_base, 'radial.rviz')],
                    output='screen',
                    condition=IfCondition(PythonExpression(["'", field_type, "' == 'radial'"]))
                ),
                # x_compress
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    arguments=['-d', os.path.join(rviz_config_base, 'x_compress.rviz')],
                    output='screen',
                    condition=IfCondition(PythonExpression(["'", field_type, "' == 'x_compress'"]))
                ),
                # y_compress
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    arguments=['-d', os.path.join(rviz_config_base, 'y_compress.rviz')],
                    output='screen',
                    condition=IfCondition(PythonExpression(["'", field_type, "' == 'y_compress'"]))
                ),
                # x_compress_tilt
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    arguments=['-d', os.path.join(rviz_config_base, 'x_compress_tilt.rviz')],
                    output='screen',
                    condition=IfCondition(PythonExpression(["'", field_type, "' == 'x_compress_tilt'"]))
                ),
                # y_compress_tilt
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    arguments=['-d', os.path.join(rviz_config_base, 'y_compress_tilt.rviz')],
                    output='screen',
                    condition=IfCondition(PythonExpression(["'", field_type, "' == 'y_compress_tilt'"]))
                ),
            ],
        ),

        # ========== POSE-AWARE PLANNER ==========

        TimerAction(
            period=5.0,  # Wait for infrastructure
            actions=[
                Node(
                    package='info_gain',
                    executable='pose_aware_planner.py',
                    name='pose_aware_planner',
                    output='screen',
                    emulate_tty=True,
                    parameters=[{
                        'field_type': field_type,
                        'trial': trial,
                        'noise_var': noise_var,
                        'lengthscale': lengthscale,
                        'horizon': horizon,
                        'lambda_cost': lambda_cost,
                        'candidate_resolution': candidate_resolution,
                        'position_std': position_std,
                        'n_mc_samples': n_mc_samples,
                    }]
                )
            ],
        ),
    ])
