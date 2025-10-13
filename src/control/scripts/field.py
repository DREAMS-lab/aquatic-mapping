#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import ColorRGBA, Float32
import numpy as np
import math


class TemperatureField(Node):
    def __init__(self):
        super().__init__('temperature_field_node')
        
        # Field parameters - 25x25 meters with 2 hotspots
        self.field_size = 25.0
        self.grid_resolution = 1.0
        self.base_temp = 20.0
        
        # Hotspots: (x, y, amplitude, sigma)
        self.hotspots = [
            (12.5, 12.5, 15.0, 7.0),    # Big central hotspot
            (18.0, 8.0, -6.0, 3.0),     # Small cold spot
        ]
        
        # Sensor noise parameters
        self.noise_sigma = 0.6 
        
        # Publishers
        self.grid_pub = self.create_publisher(Marker, '/field/visualization', 10)
        self.hotspot_pub = self.create_publisher(MarkerArray, '/field/hotspots', 10)
        self.temp_response_pub = self.create_publisher(Float32, '/field/temperature_response', 10)
        
        # Subscriber
        self.query_sub = self.create_subscription(
            PointStamped,
            '/field/query_position',
            self.query_callback,
            10
        )
        
        # Create field
        self.create_field()
        
        # Publish at 1 Hz
        self.viz_timer = self.create_timer(1.0, self.publish_visualization)
        
        self.get_logger().info(f'Temperature Field: 25x25m, 2 hotspots, noise_sigma={self.noise_sigma}°C, frame=world')
    
    def create_field(self):
        """Generate temperature field"""
        x = np.arange(0, self.field_size, self.grid_resolution)
        y = np.arange(0, self.field_size, self.grid_resolution)
        self.X, self.Y = np.meshgrid(x, y)
        
        self.temp_field = np.full_like(self.X, self.base_temp)
        
        for hx, hy, amp, sigma in self.hotspots:
            gaussian = amp * np.exp(-((self.X - hx)**2 + (self.Y - hy)**2) / (2 * sigma**2))
            self.temp_field += gaussian
    
    def get_temperature(self, x, y, add_noise=True):
        """Get temperature at (x, y)
        
        Args:
            x, y: Position coordinates
            add_noise: If True, adds Gaussian noise to simulate sensor uncertainty
        
        Returns:
            Temperature reading (°C)
        """
        temp = self.base_temp
        for hx, hy, amp, sigma in self.hotspots:
            dist_sq = (x - hx)**2 + (y - hy)**2
            temp += amp * math.exp(-dist_sq / (2 * sigma**2))
        
        if add_noise:
            # Add Gaussian noise (mean=0, std=noise_sigma)
            # Simulates real sensor uncertainty
            noise = np.random.normal(0, self.noise_sigma)
            temp += noise
        
        return temp
    
    def query_callback(self, msg):
        """temperature query"""
        x = msg.point.x
        y = msg.point.y
        temp = self.get_temperature(x, y)
        
        response = Float32()
        response.data = temp
        self.temp_response_pub.publish(response)
    
    def publish_visualization(self):
        """Publish grid and hotspots"""
        self.publish_grid()
        self.publish_hotspots()
    
    def publish_grid(self):
        """Publish temperature grid with colours"""
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'temperature_grid'
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        
        # Cube dimensions - larger and more visible
        marker.scale.x = float(self.grid_resolution)
        marker.scale.y = float(self.grid_resolution)
        marker.scale.z = 0.05
        
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Temperature range for coloring
        temp_min = float(np.min(self.temp_field))
        temp_max = float(np.max(self.temp_field))
        temp_range = max(temp_max - temp_min, 0.001)
        
        # Create grid points and colors
        rows, cols = self.X.shape
        for i in range(rows):
            for j in range(cols):
                # Position
                p = Point()
                p.x = float(self.X[i, j])
                p.y = float(self.Y[i, j])
                p.z = 0.0
                marker.points.append(p)
                
                # Color based on temperature - VIBRANT colors
                temp = float(self.temp_field[i, j])
                norm = (temp - temp_min) / temp_range
                
                color = ColorRGBA()
                if norm < 0.3:
                    # Very Cold: Deep Blue
                    color.r = 0.0
                    color.g = 0.0
                    color.b = 1.0
                elif norm < 0.5:
                    # Cold: Blue to Cyan
                    t = float((norm - 0.3) / 0.2)
                    color.r = 0.0
                    color.g = float(t)
                    color.b = 1.0
                elif norm < 0.7:
                    # Warm: Cyan to Yellow
                    t = float((norm - 0.5) / 0.2)
                    color.r = float(t)
                    color.g = 1.0
                    color.b = float(1.0 - t)
                else:
                    # Hot: Yellow to Red
                    t = float((norm - 0.7) / 0.3)
                    color.r = 1.0
                    color.g = float(1.0 - t)
                    color.b = 0.0
                
                color.a = 1.0  # Fully opaque
                marker.colors.append(color)
        
        self.grid_pub.publish(marker)
        self.get_logger().info(f'Published grid: {len(marker.points)} points', once=True)
    
    def publish_hotspots(self):
        """Publish hotspot markers"""
        array = MarkerArray()
        
        for idx, (hx, hy, amp, sigma) in enumerate(self.hotspots):
            m = Marker()
            m.header.frame_id = 'world'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'hotspots'
            m.id = idx
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            
            m.pose.position.x = float(hx)
            m.pose.position.y = float(hy)
            m.pose.position.z = 1.0
            m.pose.orientation.w = 1.0
            
            m.scale.x = float(sigma * 2)
            m.scale.y = float(sigma * 2)
            m.scale.z = 2.0
            
            # Red for hot, blue for cold
            if amp > 0:
                m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.4)
            else:
                m.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.4)
            
            array.markers.append(m)
        
        self.hotspot_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = TemperatureField()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
