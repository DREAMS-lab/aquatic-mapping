#!/usr/bin/env python3
import rclpy, numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry

class MIToPX4Relay(Node):
    def __init__(self):
        super().__init__('mi_to_px4_relay')
        
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        self.pub_offboard = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.pub_traj = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.pub_cmd = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)
        
        self.create_subscription(PoseStamped, '/mi/waypoint', self.wp_cb, 10)
        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_cb, qos)
        
        self.target = np.array([0.0, 0.0, 0.0])
        self.pos = None
        self.counter = 0
        
        self.timer = self.create_timer(0.05, self.loop)
        self.get_logger().info("PX4 relay ready")
    
    def odom_cb(self, msg):
        self.pos = np.array([msg.position[0], msg.position[1], msg.position[2]])
    
    def wp_cb(self, msg: PoseStamped):
        self.target = np.array([msg.pose.position.x, msg.pose.position.y, 0.0])
        self.get_logger().info(f"New waypoint: ({self.target[0]:.1f}, {self.target[1]:.1f})")
    
    def send_cmd(self, cmd, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.command = cmd
        msg.param1 = p1
        msg.param2 = p2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_cmd.publish(msg)
    
    def loop(self):
        om = OffboardControlMode()
        om.position = True
        om.velocity = False
        om.acceleration = False
        om.attitude = False
        om.body_rate = False
        om.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_offboard.publish(om)
        
        ts = TrajectorySetpoint()
        ts.position = [float(self.target[0]), float(self.target[1]), float(self.target[2])]
        ts.velocity = [float('nan')] * 3
        # Set yaw toward target for smoother turning
        if self.pos is not None:
            yaw = float(np.arctan2(self.target[1] - self.pos[1], self.target[0] - self.pos[0]))
        else:
            yaw = float('nan')
        ts.yaw = yaw
        ts.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_traj.publish(ts)
        
        if self.counter == 10:
            self.send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            self.get_logger().info("Armed and offboard")
        
        self.counter += 1

def main():
    rclpy.init()
    node = MIToPX4Relay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()