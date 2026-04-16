#!/usr/bin/env python3
"""
Non-Stationary Pose-Aware Planner Launch File

Launches everything needed for non-stationary pose-aware planning experiment:
  - Rover infrastructure (TF, rover_monitor, robot_state_publisher)
  - Selected field generator ONLY
  - RViz with field-specific config
  - Non-stationary pose-aware planner node (Gibbs kernel + MC expected info gain)

Usage:
  ros2 launch nonstationary_planning nonstationary_pose_aware.launch.py field_type:=radial
  ros2 launch nonstationary_planning nonstationary_pose_aware.launch.py field_type:=radial trial:=1
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    sampling_pkg_dir = get_package_share_directory('sampling')
    info_gain_pkg_dir = get_package_share_directory('info_gain')

    urdf_file = os.path.join(sampling_pkg_dir, 'urdf', 'r1_rover.urdf')
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    rviz_config_base = os.path.join(info_gain_pkg_dir, 'config', 'rviz')

    # Launch arguments
    field_type_arg = DeclareLaunchArgument('field_type', default_value='radial')
    trial_arg = DeclareLaunchArgument('trial', default_value='-1')
    lambda_cost_arg = DeclareLaunchArgument('lambda_cost', default_value='0.1')
    noise_var_arg = DeclareLaunchArgument('noise_var', default_value='0.36')
    lengthscale_arg = DeclareLaunchArgument('lengthscale', default_value='2.0')
    candidate_resolution_arg = DeclareLaunchArgument('candidate_resolution', default_value='1.0')
    position_std_arg = DeclareLaunchArgument('position_std', default_value='0.5')
    uncertainty_scale_arg = DeclareLaunchArgument('uncertainty_scale', default_value='1.0')
    n_mc_samples_arg = DeclareLaunchArgument('n_mc_samples', default_value='30')
    optimize_every_arg = DeclareLaunchArgument('optimize_every', default_value='10')
    optimize_steps_arg = DeclareLaunchArgument('optimize_steps', default_value='50')
    grid_size_arg = DeclareLaunchArgument('grid_size', default_value='5')
    l_min_arg = DeclareLaunchArgument('l_min', default_value='0.5')
    l_max_arg = DeclareLaunchArgument('l_max', default_value='5.0')

    field_type = LaunchConfiguration('field_type')
    trial = LaunchConfiguration('trial')

    return LaunchDescription([
        field_type_arg, trial_arg, lambda_cost_arg, noise_var_arg,
        lengthscale_arg, candidate_resolution_arg,
        position_std_arg, uncertainty_scale_arg, n_mc_samples_arg,
        optimize_every_arg, optimize_steps_arg,
        grid_size_arg, l_min_arg, l_max_arg,

        # === ROVER INFRASTRUCTURE ===
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='world_to_odom',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'odom'],
            output='screen',
        ),

        Node(
            package='sampling',
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

        # === FIELD GENERATORS ===
        Node(package='sampling', executable='radial_field.py', name='radial_field', output='screen',
             condition=IfCondition(PythonExpression(["'", field_type, "' == 'radial'"]))),
        Node(package='sampling', executable='x_compress_field.py', name='x_compress_field', output='screen',
             condition=IfCondition(PythonExpression(["'", field_type, "' == 'x_compress'"]))),
        Node(package='sampling', executable='y_compress_field.py', name='y_compress_field', output='screen',
             condition=IfCondition(PythonExpression(["'", field_type, "' == 'y_compress'"]))),
        Node(package='sampling', executable='x_compress_tilt_field.py', name='x_compress_tilt_field', output='screen',
             condition=IfCondition(PythonExpression(["'", field_type, "' == 'x_compress_tilt'"]))),
        Node(package='sampling', executable='y_compress_tilt_field.py', name='y_compress_tilt_field', output='screen',
             condition=IfCondition(PythonExpression(["'", field_type, "' == 'y_compress_tilt'"]))),

        # === RVIZ ===
        TimerAction(
            period=2.0,
            actions=[
                Node(package='rviz2', executable='rviz2', name='rviz2',
                     arguments=['-d', os.path.join(rviz_config_base, 'radial.rviz')], output='screen',
                     condition=IfCondition(PythonExpression(["'", field_type, "' == 'radial'"]))),
                Node(package='rviz2', executable='rviz2', name='rviz2',
                     arguments=['-d', os.path.join(rviz_config_base, 'x_compress.rviz')], output='screen',
                     condition=IfCondition(PythonExpression(["'", field_type, "' == 'x_compress'"]))),
                Node(package='rviz2', executable='rviz2', name='rviz2',
                     arguments=['-d', os.path.join(rviz_config_base, 'y_compress.rviz')], output='screen',
                     condition=IfCondition(PythonExpression(["'", field_type, "' == 'y_compress'"]))),
                Node(package='rviz2', executable='rviz2', name='rviz2',
                     arguments=['-d', os.path.join(rviz_config_base, 'x_compress_tilt.rviz')], output='screen',
                     condition=IfCondition(PythonExpression(["'", field_type, "' == 'x_compress_tilt'"]))),
                Node(package='rviz2', executable='rviz2', name='rviz2',
                     arguments=['-d', os.path.join(rviz_config_base, 'y_compress_tilt.rviz')], output='screen',
                     condition=IfCondition(PythonExpression(["'", field_type, "' == 'y_compress_tilt'"]))),
            ],
        ),

        # === NON-STATIONARY POSE-AWARE PLANNER ===
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='nonstationary_planning',
                    executable='nonstationary_pose_aware_planner.py',
                    name='nonstationary_pose_aware_planner',
                    output='screen',
                    emulate_tty=True,
                    parameters=[{
                        'field_type': field_type,
                        'trial': trial,
                        'noise_var': LaunchConfiguration('noise_var'),
                        'lengthscale': LaunchConfiguration('lengthscale'),
                        'lambda_cost': LaunchConfiguration('lambda_cost'),
                        'candidate_resolution': LaunchConfiguration('candidate_resolution'),
                        'position_std': LaunchConfiguration('position_std'),
                        'uncertainty_scale': LaunchConfiguration('uncertainty_scale'),
                        'n_mc_samples': LaunchConfiguration('n_mc_samples'),
                        'optimize_every': LaunchConfiguration('optimize_every'),
                        'optimize_steps': LaunchConfiguration('optimize_steps'),
                        'grid_size': LaunchConfiguration('grid_size'),
                        'l_min': LaunchConfiguration('l_min'),
                        'l_max': LaunchConfiguration('l_max'),
                    }]
                )
            ],
        ),
    ])
