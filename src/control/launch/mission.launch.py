from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction, LogInfo, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('control')
    urdf_file = os.path.join(pkg_dir, 'urdf', 'r1_rover.urdf')
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # RViz config
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(pkg_dir))))
    rviz_src_config = os.path.join(workspace_root, 'src', 'control', 'config', 'five_fields.rviz')
    rviz_installed_config = os.path.join(pkg_dir, 'config', 'five_fields.rviz')
    if os.path.exists(rviz_src_config):
        rviz_config = rviz_src_config
    else:
        rviz_config = rviz_installed_config

    # Declare arguments
    trial_number_arg = DeclareLaunchArgument(
        'trial_number',
        default_value='1',
        description='Trial number (1-10)'
    )
    trial_number = LaunchConfiguration('trial_number')

    return LaunchDescription([
        # TF: world -> odom
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='world_to_odom',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'odom'],
            output='screen',
        ),

        # Rover monitor (quiet) - keep TF without spam. Or delete this block to remove it entirely.
        Node(
            package='control',
            executable='rover_monitor.py',
            name='rover_monitor',
            output='log',
            emulate_tty=True,
            arguments=['--ros-args', '--log-level', 'warn']
        ),

        # Robot state publisher (delayed 1s for TF to be ready)
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

        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen'
        ),

        # ============ FIELD GENERATORS (t=0) ============
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

        # ============ SAMPLERS/RECORDERS + MISSION (t=5s) ============
        TimerAction(
            period=5.0,
            actions=[
                LogInfo(msg='Starting rosbag recorders...'),
                Node(
                    package='control',
                    executable='record_field_data.py',
                    name='radial_sampler_recorder',
                    arguments=['radial', trial_number],
                    output='screen'
                ),
                Node(
                    package='control',
                    executable='record_field_data.py',
                    name='x_compress_sampler_recorder',
                    arguments=['x_compress', trial_number],
                    output='screen'
                ),
                Node(
                    package='control',
                    executable='record_field_data.py',
                    name='x_compress_tilt_sampler_recorder',
                    arguments=['x_compress_tilt', trial_number],
                    output='screen'
                ),
                Node(
                    package='control',
                    executable='record_field_data.py',
                    name='y_compress_sampler_recorder',
                    arguments=['y_compress', trial_number],
                    output='screen'
                ),
                Node(
                    package='control',
                    executable='record_field_data.py',
                    name='y_compress_tilt_sampler_recorder',
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