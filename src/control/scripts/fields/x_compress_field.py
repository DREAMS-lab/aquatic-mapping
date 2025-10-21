#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from px4_msgs.msg import VehicleOdometry
from std_msgs.msg import Float32
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

class XCompressFieldGenerator(Node):
    def __init__(self):
        super().__init__('x_compress_field_generator')
        self.width = 25.0
        self.height = 25.0
        self.resolution = 1.0
        self.center_x = 12.5
        self.center_y = 12.5
        self.base_temperature = 20.0
        self.hotspot_amplitude = 10.0
        self.current_pos = None
        qos_viz = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        qos_px4 = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.marker_pub = self.create_publisher(Marker, '/gaussian_field/x_compress', qos_viz)
        self.temp_pub = self.create_publisher(Float32, '/gaussian_field/x_compress/temperature', 10)
        self.odom_sub = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos_px4)
        self.generate_field()
        self.viz_timer = self.create_timer(1.0, self.publish_visualization)
        self.temp_timer = self.create_timer(0.1, self.publish_temperature)
        self.get_logger().info("X-compress field initialized")
    def generate_field(self):
        sigma_x, sigma_y = 2.5, 7.0
        x = np.arange(0, self.width, self.resolution)
        y = np.arange(0, self.height, self.resolution)
        X, Y = np.meshgrid(x, y)
        gaussian = np.exp(-((X - self.center_x)**2 / (2 * sigma_x**2) + (Y - self.center_y)**2 / (2 * sigma_y**2)))
        self.temperature_field = self.base_temperature + (self.hotspot_amplitude * gaussian)
        self.X, self.Y = X, Y
        self.temp_min = float(np.min(self.temperature_field))
        self.temp_max = float(np.max(self.temperature_field))
        self.get_logger().info(f"X-compress field: [{self.temp_min:.2f}, {self.temp_max:.2f}]°C")
    def sample_field(self, x, y):
        if x < 0 or x > self.width or y < 0 or y > self.height:
            return self.base_temperature
        x_grid = np.arange(0, self.width, self.resolution)
        y_grid = np.arange(0, self.height, self.resolution)
        i = np.searchsorted(x_grid, x) - 1
        j = np.searchsorted(y_grid, y) - 1
        if i < 0 or i >= len(x_grid)-1 or j < 0 or j >= len(y_grid)-1:
            return self.base_temperature
        x0, x1 = x_grid[i], x_grid[i+1]
        y0, y1 = y_grid[j], y_grid[j+1]
        wx = (x - x0) / (x1 - x0)
        wy = (y - y0) / (y1 - y0)
        temp = (1-wx)*(1-wy)*self.temperature_field[j,i] + wx*(1-wy)*self.temperature_field[j,i+1] + (1-wx)*wy*self.temperature_field[j+1,i] + wx*wy*self.temperature_field[j+1,i+1]
        return float(temp)
    def odom_callback(self, msg):
        self.current_pos = np.array([msg.position[0], msg.position[1], msg.position[2]])
    def publish_temperature(self):
        if self.current_pos is None:
            return
        temp = self.sample_field(self.current_pos[0], self.current_pos[1])
        msg = Float32()
        msg.data = temp
        self.temp_pub.publish(msg)
    def publish_visualization(self):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "x_compress_field"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.scale.x = self.resolution
        marker.scale.y = self.resolution
        marker.scale.z = 0.1
        marker.pose.orientation.w = 1.0
        rows, cols = self.X.shape
        for i in range(rows):
            for j in range(cols):
                temp = self.temperature_field[i, j]
                p = Point()
                p.x, p.y, p.z = float(self.X[i, j]), float(self.Y[i, j]), -0.2
                marker.points.append(p)
                normalized = (temp - self.temp_min) / (self.temp_max - self.temp_min)
                color = ColorRGBA()
                color.a = 0.8
                color.r = 0.5 + normalized * 0.5
                color.g = normalized * 0.5
                color.b = 1.0 - normalized * 0.5
                marker.colors.append(color)
        self.marker_pub.publish(marker)
def main(args=None):
    rclpy.init(args=args)
    node = XCompressFieldGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()