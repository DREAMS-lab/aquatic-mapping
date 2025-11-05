#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleOdometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

class Px4OdomTF(Node):
    def __init__(self):
        super().__init__('px4_odom_tf')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.br = TransformBroadcaster(self)
        self.sub = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.cb, qos)
        self.get_logger().info("PX4 odom->TF broadcaster ready")

    def cb(self, msg: VehicleOdometry):
        # PX4 VehicleOdometry.q is [w, x, y, z]
        qw, qx, qy, qz = float(msg.q[0]), float(msg.q[1]), float(msg.q[2]), float(msg.q[3])
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'       # you already have world->odom static
        t.child_frame_id = 'base_link'  # match your URDF root
        t.transform.translation.x = float(msg.position[0])
        t.transform.translation.y = float(msg.position[1])
        t.transform.translation.z = float(msg.position[2])
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.br.sendTransform(t)

def main():
    rclpy.init()
    n = Px4OdomTF()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

