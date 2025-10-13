#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry
import numpy as np


class SimpleWaypointSender(Node):
    def __init__(self):
        super().__init__('simple_waypoint_sender')
        
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
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.counter = 0
        self.position_received = False
        
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('Waypoint Sender: Ready')

    def odometry_callback(self, msg):
        self.current_position = np.array([msg.position[0], msg.position[1], msg.position[2]])
        
        if not self.position_received:
            x, y = self.current_position[0], self.current_position[1]
            self.get_logger().info(f'Current position: X={x:.2f}m, Y={y:.2f}m')
            self.position_received = True

    def control_loop(self):
        offboard_msg = OffboardControlMode()
        offboard_msg.position = True
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_pub.publish(offboard_msg)
        
        setpoint_msg = TrajectorySetpoint()
        setpoint_msg.position = [
            float(self.target_position[0]),
            float(self.target_position[1]),
            float(self.target_position[2])
        ]
        setpoint_msg.velocity = [float('nan')] * 3
        setpoint_msg.yaw = float('nan')
        setpoint_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_pub.publish(setpoint_msg)
        
        if self.counter == 10:
            self.engage_offboard_mode()
            self.arm()
            self.get_logger().info('Armed and in offboard mode')
        
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

    def disarm(self):
        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)

    def engage_offboard_mode(self):
        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)

    def go_to(self, x, y):
        self.target_position = np.array([x, y, 0.0])
        self.get_logger().info(f'Target: X={x}m, Y={y}m')


def main(args=None):
    rclpy.init(args=args)
    node = SimpleWaypointSender()
    
    import threading
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()
    
    import time
    time.sleep(2)
    
    try:
        while rclpy.ok():
            time.sleep(0.1)
            user_input = input("Enter X Y: ").strip()
            
            if user_input.lower() in ['quit', 'q', 'exit']:
                break
            
            try:
                parts = user_input.split()
                if len(parts) != 2:
                    print("Enter 2 numbers")
                    continue
                
                x = float(parts[0])
                y = float(parts[1])
                
                node.go_to(x, y)
                print(f"Going to ({x}, {y})")
                
            except ValueError:
                print("Invalid input")
            except EOFError:
                break
    
    except KeyboardInterrupt:
        pass
    
    finally:
        print("\nShutting down...")
        try:
            node.disarm()
        except:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()