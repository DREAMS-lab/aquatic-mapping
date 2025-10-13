#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleOdometry
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import threading
import time


class TemperatureQuery(Node):
    def __init__(self):
        super().__init__('temp_query')
        
        # QoS profile for PX4 messages
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribe to rover position
        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            qos_profile
        )
        
        # Subscribe to temperature response
        self.temp_sub = self.create_subscription(
            Float32,
            '/field/temperature_response',
            self.temp_callback,
            10
        )
        
        # Publisher for temperature query
        self.query_pub = self.create_publisher(PointStamped, '/field/query_position', 10)
        
        # Current position
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_temp = None
        self.position_received = False
        
        self.get_logger().info('working')
    
    def odom_callback(self, msg):
        """Get current rover position"""
        self.current_x = msg.position[0]
        self.current_y = msg.position[1]
        if not self.position_received:
            self.position_received = True
    
    def temp_callback(self, msg):
        """Receive temperature response"""
        self.current_temp = msg.data
    
    def query_temperature(self):
        """Query temperature at current rover position"""
        if not self.position_received:
            print("Waiting for rover position...")
            return
        
        query = PointStamped()
        query.header.stamp = self.get_clock().now().to_msg()
        query.header.frame_id = 'odom'
        query.point.x = float(self.current_x)
        query.point.y = float(self.current_y)
        query.point.z = 0.0
        
        self.query_pub.publish(query)
        
        # Wait briefly for response
        time.sleep(0.1)
        
        if self.current_temp is not None:
            print(f"Position: ({self.current_x:.2f}, {self.current_y:.2f}) | Temperature: {self.current_temp:.2f}°C")
        else:
            print(f"Position: ({self.current_x:.2f}, {self.current_y:.2f}) | Temperature: No data")


def main(args=None):
    rclpy.init(args=args)
    node = TemperatureQuery()
    
    # Start ROS spinning in background
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()
    
    # Wait for position
    time.sleep(1)

    
    try:
        while rclpy.ok():
            user_input = input("").strip()
            
            if user_input.lower() in ['quit', 'q', 'exit']:
                break
            
            # Query on any input (including just pressing enter)
            node.query_temperature()
    
    except KeyboardInterrupt:
        pass
    except EOFError:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()