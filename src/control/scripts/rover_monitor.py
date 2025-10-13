#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleOdometry
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformBroadcaster
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class RoverMonitor(Node):
    def __init__(self):
        super().__init__('rover_monitor')
        
        # QoS profile for PX4 messages
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribe to PX4 odometry
        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            qos_profile
        )
        
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Marker publisher for RViz2
        self.marker_pub = self.create_publisher(Marker, '/rover/marker', 10)
        
        # Boundary limits
        self.MAX_X = 100.0
        self.MAX_Y = 100.0
        self.MIN_X = -10.0
        self.MIN_Y = -10.0
        
        # Current position
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        
        # Message counter for periodic logging
        self.msg_count = 0
        
        self.get_logger().info('Rover Monitor: Ready')
    
    def odom_callback(self, msg):
        """Process odometry data and publish TF and marker"""
        # Extract actual position from odometry
        self.current_x = msg.position[0]
        self.current_y = msg.position[1]
        self.current_z = msg.position[2]
        
        # Check boundaries
        self.check_boundaries()
        
        # Publish TF
        self.publish_tf(msg)
        
        # Publish marker
        self.publish_marker()
        
        # Minimal periodic logging (every 20 messages ~ 1 Hz)
        self.msg_count += 1
        if self.msg_count % 20 == 0:
            self.get_logger().info(
                f'Position: X={self.current_x:6.2f}m, Y={self.current_y:6.2f}m'
            )
    
    def check_boundaries(self):
        """Check if rover is within safe boundaries"""
        if (self.current_x > self.MAX_X or self.current_x < self.MIN_X or 
            self.current_y > self.MAX_Y or self.current_y < self.MIN_Y):
            self.get_logger().warn(
                f'BOUNDARY WARNING: ({self.current_x:.2f}, {self.current_y:.2f})',
                throttle_duration_sec=2.0
            )
    
    def publish_tf(self, odom_msg):
        """Publish transform from odom to base_link"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        
        # Position
        t.transform.translation.x = float(self.current_x)
        t.transform.translation.y = float(self.current_y)
        t.transform.translation.z = float(self.current_z)
        
        # Orientation
        t.transform.rotation.w = float(odom_msg.q[0])
        t.transform.rotation.x = float(odom_msg.q[1])
        t.transform.rotation.y = float(odom_msg.q[2])
        t.transform.rotation.z = float(odom_msg.q[3])
        
        self.tf_broadcaster.sendTransform(t)
    
    def publish_marker(self):
        """Publish visual marker for RViz2"""
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rover"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.15
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.5
        marker.scale.y = 0.4
        marker.scale.z = 0.3
        
        marker.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.9)
        
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = RoverMonitor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()