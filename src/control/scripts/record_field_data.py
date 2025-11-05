#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleOdometry
from std_msgs.msg import Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import subprocess
import os
import sys
import shutil
import csv
import numpy as np
import signal


class FieldDataRecorder(Node):
    def __init__(self, field_name, trial_number):
        super().__init__(f'{field_name}_recorder')
        
        self.field_name = field_name
        self.trial_number = trial_number
        
        # Setup paths
        workspace_root = '/home/dreams-lab-u24/workspaces/aquatic-mapping'
        trial_folder = os.path.join(workspace_root, 'src', 'control', 'data', 
                                    field_name, f'trial_{trial_number}')
        os.makedirs(trial_folder, exist_ok=True)
        
        self.bag_path = os.path.join(trial_folder, f'{field_name}_bag')
        self.csv_path = os.path.join(trial_folder, f'{field_name}_samples.csv')
        
        # Delete old files
        if os.path.exists(self.bag_path):
            shutil.rmtree(self.bag_path)
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        
        # Create CSV with header and keep file open
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['x', 'y', 'temperature'])
        
        # Waypoints for start/stop detection
        self.first_waypoint = np.array([2.5, 2.5])  # Start of mission
        self.last_waypoint = np.array([0.0, 0.0])   # End of mission
        self.waypoint_threshold = 0.5  # Within 0.5m
        
        # State variables
        self.current_pos = None
        self.current_temp = None
        self.recording_started = False
        self.recording_stopped = False
        self.recording_process = None
        self.csv_rows_written = 0
        
        # QoS for PX4
        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribe to odometry
        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            qos_px4
        )
        
        # Subscribe to temperature
        self.temp_sub = self.create_subscription(
            Float32,
            f'/gaussian_field/{field_name}/temperature',
            self.temp_callback,
            10
        )
        
        # Timer to record every 1 second
        self.record_timer = self.create_timer(1.0, self.record_data_point)
        
        self.get_logger().info(f'Recorder initialized: {field_name}, trial {trial_number}')
        self.get_logger().info(f'Waiting for rover to reach first waypoint ({self.first_waypoint[0]}, {self.first_waypoint[1]})...')
    
    def odom_callback(self, msg):
        """Update position and handle start/stop based on waypoints."""
        self.current_pos = np.array([msg.position[0], msg.position[1]])
        
        # Check if at first waypoint - START recording
        if not self.recording_started:
            dist_to_first = np.linalg.norm(self.current_pos - self.first_waypoint)
            if dist_to_first < self.waypoint_threshold:
                self.start_recording()
        
        # Check if at last waypoint - STOP recording
        elif self.recording_started and not self.recording_stopped:
            dist_to_last = np.linalg.norm(self.current_pos - self.last_waypoint)
            if dist_to_last < self.waypoint_threshold:
                self.stop_recording()
    
    def temp_callback(self, msg):
        """Update current temperature."""
        self.current_temp = msg.data
    
    def record_data_point(self):
        """Record data every 1 second if recording active."""
        if not self.recording_started or self.recording_stopped:
            return
        
        if self.current_pos is None or self.current_temp is None:
            return
        
        # Write to CSV
        self.csv_writer.writerow([
            float(self.current_pos[0]),
            float(self.current_pos[1]),
            float(self.current_temp)
        ])
        self.csv_file.flush()  # Force write to disk
        
        self.csv_rows_written += 1
        
        if self.csv_rows_written % 20 == 0:
            self.get_logger().info(f'Recorded {self.csv_rows_written} points')
    
    def start_recording(self):
        """Start recording."""
        if self.recording_started:
            return
        
        self.recording_started = True
        self.get_logger().info('=' * 60)
        self.get_logger().info('Recording STARTED - reached first waypoint')
        self.get_logger().info('=' * 60)
        
        # Start rosbag
        cmd = [
            'ros2', 'bag', 'record',
            '-o', self.bag_path,
            '/fmu/out/vehicle_odometry',
            f'/gaussian_field/{self.field_name}/temperature'
        ]
        
        try:
            self.recording_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )
        except Exception as e:
            self.get_logger().error(f'Bag start failed: {e}')
    
    def stop_recording(self):
        """Stop recording."""
        if self.recording_stopped:
            return
        
        self.recording_stopped = True
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('Recording STOPPED - reached final waypoint')
        self.get_logger().info('=' * 60)
        
        if self.recording_process and self.recording_started:
            try:
                os.killpg(os.getpgid(self.recording_process.pid), signal.SIGINT)
                self.recording_process.wait(timeout=5)
            except:
                self.recording_process.kill()
            
            self.get_logger().info(f'CSV: {self.csv_path} ({self.csv_rows_written} points)')
            self.get_logger().info(f'Bag: {self.bag_path}')

    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'csv_file'):
                self.csv_file.close()
        except Exception:
            pass


def main(args=None):
    if len(sys.argv) < 3:
        print("\nUsage: ros2 run control record_field_data.py <field_name> <trial_number>\n")
        return
    
    field_name = sys.argv[1]
    trial_number = sys.argv[2]
    
    rclpy.init(args=args)
    node = FieldDataRecorder(field_name, int(trial_number))
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not node.recording_stopped:
            node.stop_recording()
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()