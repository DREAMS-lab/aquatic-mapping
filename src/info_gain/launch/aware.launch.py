#!/usr/bin/env python3
"""
Uncertainty-Aware Sampler Launch File (Complete)

Launches everything needed for uncertainty-aware experiment:
  - Rover infrastructure (TF, rover_monitor, robot_state_publisher, RViz)
  - Selected field generator ONLY
  - Aware sampler node (Code B: accounts for positional uncertainty via Monte Carlo)

Usage:
  ros2 launch info_gain aware.launch.py field_type:=radial

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
    # Get sampling package directory for URDF and RViz config
    sampling_pkg_dir = get_package_share_directory('sampling')

    # Load URDF
    urdf_file = os.path.join(sampling_pkg_dir, 'urdf', 'r1_rover.urdf')
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # RViz config
    rviz_config = os.path.join(sampling_pkg_dir, 'config', 'rviz', 'rover.rviz')

    # Declare launch arguments
    field_type_arg = DeclareLaunchArgument(
        'field_type',
        default_value='radial',
        description='Field type to sample (radial, x_compress, y_compress, x_compress_tilt, y_compress_tilt)'
    )

    n_initial_arg = DeclareLaunchArgument(
        'n_initial',
        default_value='5',
        description='Number of initial samples (sparse coverage)'
    )

    n_adaptive_arg = DeclareLaunchArgument(
        'n_adaptive',
        default_value='30',
        description='Number of adaptive samples'
    )

    noise_var_arg = DeclareLaunchArgument(
        'noise_var',
        default_value='0.01',
        description='Observation noise variance'
    )

    lengthscale_arg = DeclareLaunchArgument(
        'lengthscale',
        default_value='2.0',
        description='GP lengthscale'
    )

    top_k_arg = DeclareLaunchArgument(
        'top_k',
        default_value='5',
        description='Top-K selection parameter'
    )

    sigma_x_arg = DeclareLaunchArgument(
        'sigma_x',
        default_value='0.5',
        description='Default positional std_x (overridden by odometry covariance)'
    )

    sigma_y_arg = DeclareLaunchArgument(
        'sigma_y',
        default_value='0.5',
        description='Default positional std_y (overridden by odometry covariance)'
    )

    mc_samples_arg = DeclareLaunchArgument(
        'mc_samples',
        default_value='20',
        description='Number of Monte Carlo samples for expected information gain'
    )

    max_steps_arg = DeclareLaunchArgument(
        'max_steps',
        default_value='100',
        description='Hard budget cap on adaptive samples'
    )

    eps_info_arg = DeclareLaunchArgument(
        'eps_info',
        default_value='0.001',
        description='Information gain saturation threshold'
    )

    patience_arg = DeclareLaunchArgument(
        'patience',
        default_value='5',
        description='Consecutive low-info steps before stopping'
    )

    # Get launch configurations
    field_type = LaunchConfiguration('field_type')
    n_initial = LaunchConfiguration('n_initial')
    n_adaptive = LaunchConfiguration('n_adaptive')
    noise_var = LaunchConfiguration('noise_var')
    lengthscale = LaunchConfiguration('lengthscale')
    top_k = LaunchConfiguration('top_k')
    sigma_x = LaunchConfiguration('sigma_x')
    sigma_y = LaunchConfiguration('sigma_y')
    mc_samples = LaunchConfiguration('mc_samples')
    max_steps = LaunchConfiguration('max_steps')
    eps_info = LaunchConfiguration('eps_info')
    patience = LaunchConfiguration('patience')

    return LaunchDescription([
        # ========== LAUNCH ARGUMENTS ==========
        field_type_arg,
        n_initial_arg,
        n_adaptive_arg,
        noise_var_arg,
        lengthscale_arg,
        top_k_arg,
        sigma_x_arg,
        sigma_y_arg,
        mc_samples_arg,
        max_steps_arg,
        eps_info_arg,
        patience_arg,

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

        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen'
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

        # ========== UNCERTAINTY-AWARE SAMPLER ==========

        TimerAction(
            period=3.0,  # Wait for infrastructure to initialize
            actions=[
                Node(
                    package='info_gain',
                    executable='aware_sampler.py',
                    name='aware_sampler',
                    output='screen',
                    emulate_tty=True,
                    parameters=[{
                        'field_type': field_type,
                        'n_initial': n_initial,
                        'n_adaptive': n_adaptive,
                        'noise_var': noise_var,
                        'lengthscale': lengthscale,
                        'top_k': top_k,
                        'sigma_x': sigma_x,
                        'sigma_y': sigma_y,
                        'mc_samples': mc_samples,
                        'max_steps': max_steps,
                        'eps_info': eps_info,
                        'patience': patience,
                    }]
                )
            ],
        ),
    ])
