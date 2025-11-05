#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

class WorldFramePublisher(Node):
    def __init__(self):
        super().__init__('world_frame_publisher')
        
        # Create static transform broadcaster
        self.static_broadcaster = StaticTransformBroadcaster(self)
        
        # Publish the world frame (root of TF tree)
        self.publish_world_frame()
        
        self.get_logger().info("World frame publisher initialized")
        self.get_logger().info("Publishing static transform: world (root frame)")
    
    def publish_world_frame(self):
        """Publish static identity transform to establish world as root frame."""
        static_transform = TransformStamped()
        
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = "world"
        static_transform.child_frame_id = "map"  # Publish map as child of world
        
        # Identity transform
        static_transform.transform.translation.x = 0.0
        static_transform.transform.translation.y = 0.0
        static_transform.transform.translation.z = 0.0
        
        static_transform.transform.rotation.x = 0.0
        static_transform.transform.rotation.y = 0.0
        static_transform.transform.rotation.z = 0.0
        static_transform.transform.rotation.w = 1.0
        
        # Publish the static transform
        self.static_broadcaster.sendTransform(static_transform)
        
        self.get_logger().info("World frame established (world -> map) in TF tree")

def main(args=None):
    rclpy.init(args=args)
    publisher = WorldFramePublisher()
    
    publisher.get_logger().info("World frame publisher running. Press Ctrl+C to exit.")
    
    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        publisher.get_logger().info("Node interrupted by user.")
    finally:
        publisher.get_logger().info("Shutting down node.")
        try:
            publisher.destroy_node()
        except Exception as e:
            print(f"Node cleanup failed: {e}")
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

