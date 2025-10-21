#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from px4_msgs.msg import VehicleOdometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import threading


class MultiFieldQuery(Node):
    def __init__(self):
        super().__init__('multi_field_query')
        
        # Define field display info
        self.fields = {
            'radial': {'name': 'Radial', 'color': '\033[96m', 'temp': None},
            'x_compress': {'name': 'X-Compress', 'color': '\033[95m', 'temp': None},
            'x_compress_tilt': {'name': 'X-Compress-Tilt', 'color': '\033[92m', 'temp': None},
            'y_compress': {'name': 'Y-Compress', 'color': '\033[94m', 'temp': None},
            'y_compress_tilt': {'name': 'Y-Compress-Tilt', 'color': '\033[93m', 'temp': None},
        }
        self.reset_color = '\033[0m'
        
        # Subscribe to temperature topics from all 5 field generators
        for field_key in self.fields.keys():
            self.create_subscription(
                Float32,
                f'/gaussian_field/{field_key}/temperature',
                lambda msg, key=field_key: self.temp_callback(msg, key),
                10
            )
        
        # Subscribe to rover position for display
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            qos_profile
        )
        
        # Current position
        self.current_x = 12.5
        self.current_y = 12.5
        self.position_received = False
        
        self.get_logger().info('Multi-Field Query initialized')
        self.get_logger().info('Subscribing to temperature values from 5 field generators')
    
    def temp_callback(self, msg, field_key):
        """Callback for temperature updates from field generators."""
        self.fields[field_key]['temp'] = msg.data
    
    def odom_callback(self, msg):
        """Get current rover position from PX4 odometry."""
        self.current_x = msg.position[0]
        self.current_y = msg.position[1]
        if not self.position_received:
            self.position_received = True
    
    def display_all_fields(self):
        """Display all field temperature values."""
        if not self.position_received:
            print("\nWaiting for rover position from /fmu/out/vehicle_odometry...")
            print("(Make sure PX4 simulation is running)")
            return
        
        print(f"\nRover Position: ({self.current_x:.2f}, {self.current_y:.2f}) m\n")
        
        for field_key, field_info in self.fields.items():
            temp = field_info['temp']
            name = field_info['name']
            color = field_info['color']
            
            if temp is not None:
                print(f"{color}  {name:20s}: {temp:6.2f} °C{self.reset_color}")
            else:
                print(f"  {name:20s}: Waiting for data...")
        
        print("")


def main(args=None):
    rclpy.init(args=args)
    node = MultiFieldQuery()
    
    # Start ROS spinning in background
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()
    
    # Wait for initial data
    import time
    time.sleep(1)
    
    print("\nMulti-Field Temperature Query")
    print("Press ENTER to display all 5 field temperatures at rover position")
    print("Type 'q' or 'quit' to exit\n")
    
    try:
        while rclpy.ok():
            user_input = input("").strip()
            
            if user_input.lower() in ['quit', 'q', 'exit']:
                print("\nExiting...\n")
                break
            
            # Display on any input (including just pressing enter)
            node.display_all_fields()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted. Exiting...\n")
    except EOFError:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
