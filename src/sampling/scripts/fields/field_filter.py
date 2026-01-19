#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleOdometry
from std_msgs.msg import Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import sys

class FieldFilter(Node):
    """Filter that republishes at GRID POINTS (every 1m) inside mission boundary."""
    
    def __init__(self, field_name):
        super().__init__(f'{field_name}_filter')
        self.field_name = field_name
        
        # Mission boundary
        self.x_min = 2.5
        self.x_max = 22.5
        self.y_min = 2.5
        self.y_max = 22.5
        
        # Grid parameters
        self.grid_spacing = 1.0  # 1 meter grid
        self.snap_threshold = 0.3  # Within 0.3m of grid point to trigger
        
        self.recorded_grid_points = set()  # Track which grid points we've recorded
        self.current_pos = None
        self.current_temp = None
        self.current_odom = None
        self.points_published = 0
        self.mission_started = False
        self.mission_ended = False
        
        # QoS
        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribe to raw topics
        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            qos_px4
        )
        
        self.temp_sub = self.create_subscription(
            Float32,
            f'/gaussian_field/{field_name}/temperature',
            self.temp_callback,
            10
        )
        
        # Publish filtered topics
        self.odom_pub = self.create_publisher(VehicleOdometry, f'/filtered/{field_name}/odometry', 10)
        self.temp_pub = self.create_publisher(Float32, f'/filtered/{field_name}/temperature', 10)
        
        self.get_logger().info(f'{field_name} filter initialized - grid-based recording (1m spacing)')
        self.get_logger().info(f'Mission boundary: X[{self.x_min}, {self.x_max}], Y[{self.y_min}, {self.y_max}]')
    
    def is_inside_boundary(self, pos):
        """Check if position is inside mission boundary."""
        return (self.x_min <= pos[0] <= self.x_max and 
                self.y_min <= pos[1] <= self.y_max)
    
    def get_nearest_grid_point(self, x, y):
        """Get nearest grid point (rounded to nearest meter)."""
        grid_x = round(x / self.grid_spacing) * self.grid_spacing
        grid_y = round(y / self.grid_spacing) * self.grid_spacing
        return (grid_x, grid_y)
    
    def distance_to_grid_point(self, x, y, grid_x, grid_y):
        """Calculate distance to a grid point."""
        return np.sqrt((x - grid_x)**2 + (y - grid_y)**2)
    
    def odom_callback(self, msg):
        self.current_pos = np.array([msg.position[0], msg.position[1]])
        self.current_odom = msg
        
        current_inside = self.is_inside_boundary(self.current_pos)
        
        # Detect mission start
        if current_inside and not self.mission_started:
            self.mission_started = True
            self.get_logger().info('='*60)
            self.get_logger().info('🟢 MISSION STARTED - Recording at grid points')
            self.get_logger().info('='*60)
        
        # Detect mission end
        if not current_inside and self.mission_started and not self.mission_ended:
            self.mission_ended = True
            self.get_logger().info('='*60)
            self.get_logger().info('🔴 MISSION ENDED')
            self.get_logger().info(f'Total grid points collected: {self.points_published}')
            self.get_logger().info('='*60)
        
        self.try_publish()
    
    def temp_callback(self, msg):
        self.current_temp = msg
    
    def try_publish(self):
        """Publish when rover is near a grid point we haven't recorded yet."""
        if self.current_pos is None or self.current_temp is None or self.current_odom is None:
            return
        
        # Don't publish if mission hasn't started or has ended
        if not self.mission_started or self.mission_ended:
            return
        
        # Only publish if inside mission boundary
        if not self.is_inside_boundary(self.current_pos):
            return
        
        # Get nearest grid point
        grid_x, grid_y = self.get_nearest_grid_point(self.current_pos[0], self.current_pos[1])
        grid_key = (grid_x, grid_y)
        
        # Check if we're close to this grid point
        distance = self.distance_to_grid_point(self.current_pos[0], self.current_pos[1], grid_x, grid_y)
        
        # If we're within threshold AND haven't recorded this grid point yet
        if distance < self.snap_threshold and grid_key not in self.recorded_grid_points:
            self.recorded_grid_points.add(grid_key)
            self.publish(grid_x, grid_y)
    
    def publish(self, grid_x, grid_y):
        """Publish at grid coordinates."""
        # Publish original odometry
        self.odom_pub.publish(self.current_odom)
        self.temp_pub.publish(self.current_temp)
        
        self.points_published += 1
        self.get_logger().info(f'📍 Point {self.points_published}: Grid({grid_x:.0f}, {grid_y:.0f}) = {self.current_temp.data:.2f}°C')


def main(args=None):
    if len(sys.argv) < 2:
        print("Usage: ros2 run control field_filter.py <field_name>")
        return
    
    field_name = sys.argv[1]
    rclpy.init(args=args)
    node = FieldFilter(field_name)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()