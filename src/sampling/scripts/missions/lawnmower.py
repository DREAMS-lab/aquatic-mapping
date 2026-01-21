#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry
import numpy as np
import time
import os
import sys


class LawnmowerMission(Node):
    def __init__(self):
        super().__init__('lawnmower_mission')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)

        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)

        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        self.odometry_sub = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self.odometry_callback, qos_profile)

        self.current_position = np.array([0.0, 0.0, 0.0])
        self.counter = 0
        self.waypoint_idx = 0
        self.mission_started = False
        self.mission_complete = False
        self.start_time = None

        # Auto-shutdown configuration
        # Set AUTO_STOP=0 to disable auto-shutdown after mission complete
        self.auto_stop = os.environ.get('AUTO_STOP', '1') == '1'
        self.shutdown_delay = int(os.environ.get('SHUTDOWN_DELAY', '10'))  # seconds after mission complete
        self.shutdown_counter = None
        self.shutdown_requested = False

        # Generate lawnmower waypoints
        self.waypoints = self.generate_lawnmower_waypoints()
        self.current_target = self.waypoints[0]

        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info(f'Lawnmower Mission: {len(self.waypoints)} waypoints generated')
        if self.auto_stop:
            self.get_logger().info(f'Auto-stop enabled: will shutdown {self.shutdown_delay}s after mission complete')

    def generate_lawnmower_waypoints(self):
        """Generate lawnmower pattern with 2.5m border buffer and 5m row spacing."""
        waypoints = []
        
        # Start at origin (outside field)
        waypoints.append(np.array([0.0, 0.0, 0.0]))
        
        # Move to first corner of inner square (2.5m buffer)
        waypoints.append(np.array([2.5, 2.5, 0.0]))
        
        # Lawnmower parameters
        row_spacing = 5.0  # 5m between rows (was 1.0)
        x_min = 2.5  # 2.5m from left edge
        x_max = 22.5  # 2.5m from right edge (25 - 2.5)
        y_min = 2.5  # 2.5m from bottom edge
        y_max = 22.5  # 2.5m from top edge (25 - 2.5)
        
        current_y = y_min
        going_east = True
        
        while current_y <= y_max:
            if going_east:
                # Go east to x_max
                waypoints.append(np.array([x_max, current_y, 0.0]))
                going_east = False
            else:
                # Go west to x_min
                waypoints.append(np.array([x_min, current_y, 0.0]))
                going_east = True
            
            # Move north by row_spacing
            current_y += row_spacing
            
            if current_y <= y_max:
                if going_east:
                    waypoints.append(np.array([x_min, current_y, 0.0]))
                else:
                    waypoints.append(np.array([x_max, current_y, 0.0]))
        
        # Return to origin
        waypoints.append(np.array([0.0, 0.0, 0.0]))
        
        return waypoints

    def odometry_callback(self, msg):
        self.current_position = np.array([msg.position[0], msg.position[1], msg.position[2]])

    def control_loop(self):
        # Publish offboard control mode
        offboard_msg = OffboardControlMode()
        offboard_msg.position = True
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_pub.publish(offboard_msg)
        
        # Publish trajectory setpoint
        setpoint_msg = TrajectorySetpoint()
        setpoint_msg.position = [
            float(self.current_target[0]),
            float(self.current_target[1]),
            float(self.current_target[2])
        ]
        setpoint_msg.velocity = [float('nan')] * 3
        setpoint_msg.yaw = float('nan')
        setpoint_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_pub.publish(setpoint_msg)
        
        # Arm and engage offboard after 10 iterations
        if self.counter == 10:
            self.engage_offboard_mode()
            self.arm()
            self.start_time = time.time()
            self.get_logger().info('Armed and in offboard mode. Holding at (0,0) for 3 seconds...')
        
        # Wait 3 seconds at start position before starting mission
        if self.counter > 10 and not self.mission_started:
            if time.time() - self.start_time >= 3.0:
                self.mission_started = True
                self.get_logger().info('Starting lawnmower mission!')
        
        # Mission logic: advance waypoints when close enough
        if self.mission_started and self.waypoint_idx < len(self.waypoints) - 1:
            distance = np.linalg.norm(self.current_position[:2] - self.current_target[:2])
            
            if distance < 0.5:  # Within 0.5m of waypoint
                self.waypoint_idx += 1
                self.current_target = self.waypoints[self.waypoint_idx]
                self.get_logger().info(
                    f'Waypoint {self.waypoint_idx}/{len(self.waypoints)}: '
                    f'({self.current_target[0]:.1f}, {self.current_target[1]:.1f})'
                )
        
        # Mission complete - hold position and optionally auto-shutdown
        if self.waypoint_idx >= len(self.waypoints) - 1 and not self.mission_complete:
            distance = np.linalg.norm(self.current_position[:2] - self.current_target[:2])
            if distance < 0.5:
                self.mission_complete = True
                self.get_logger().info('='*60)
                self.get_logger().info('MISSION COMPLETE! Returned to (0, 0)')
                if self.auto_stop:
                    self.get_logger().info(f'Auto-shutdown in {self.shutdown_delay} seconds...')
                    self.shutdown_counter = self.shutdown_delay * 20  # 20 Hz timer
                else:
                    self.get_logger().info('Holding position. Press Ctrl+C to stop.')
                self.get_logger().info('='*60)

        # Handle auto-shutdown countdown
        if self.shutdown_counter is not None and not self.shutdown_requested:
            self.shutdown_counter -= 1
            if self.shutdown_counter <= 0:
                self.shutdown_requested = True
                self.get_logger().info('AUTO-STOP: Shutting down ROS node...')
                # Request shutdown - will be handled in main()
                raise SystemExit(0)

        self.counter += 1

    def send_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(msg)

    def arm(self):
        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)

    def engage_offboard_mode(self):
        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)


def main(args=None):
    rclpy.init(args=args)
    node = LawnmowerMission()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    except SystemExit as e:
        # Auto-shutdown triggered
        node.get_logger().info('Mission auto-shutdown complete')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)


if __name__ == '__main__':
    main()