#!/usr/bin/env python3
import csv
import time
import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32
from px4_msgs.msg import VehicleOdometry
from ament_index_python.packages import get_package_share_directory

NEAR_RADIUS = 1.5

def ned_to_enu(p_xyz):
    """PX4 -> ROS: (x_north, y_east, z_down) -> (x_east, y_north, z_up)"""
    x_n, y_e, z_d = float(p_xyz[0]), float(p_xyz[1]), float(p_xyz[2])
    return np.array([y_e, x_n, -z_d], dtype=float)

class SampleLogger(Node):
    def __init__(self):
        super().__init__('sample_logger')

        self.sub_goal = self.create_subscription(
            PointStamped, '/sampling/commanded_location', self.cb_goal, 10)

        # QoS profile for PX4 messages
        qos_px4 = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub_odom = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.cb_odom, qos_px4)

        # Subscribe to all fields - clean versions
        self.sub_radial = self.create_subscription(
            Float32, '/gaussian_field/radial/temperature', self.cb_radial, 10)
        self.sub_x_compress = self.create_subscription(
            Float32, '/gaussian_field/x_compress/temperature', self.cb_x_compress, 10)
        self.sub_y_compress = self.create_subscription(
            Float32, '/gaussian_field/y_compress/temperature', self.cb_y_compress, 10)
        self.sub_x_compress_tilt = self.create_subscription(
            Float32, '/gaussian_field/x_compress_tilt/temperature', self.cb_x_compress_tilt, 10)
        self.sub_y_compress_tilt = self.create_subscription(
            Float32, '/gaussian_field/y_compress_tilt/temperature', self.cb_y_compress_tilt, 10)

        # Subscribe to all fields - noisy versions
        self.sub_radial_noisy = self.create_subscription(
            Float32, '/gaussian_field/radial/temperature_noisy', self.cb_radial_noisy, 10)
        self.sub_x_compress_noisy = self.create_subscription(
            Float32, '/gaussian_field/x_compress/temperature_noisy', self.cb_x_compress_noisy, 10)
        self.sub_y_compress_noisy = self.create_subscription(
            Float32, '/gaussian_field/y_compress/temperature_noisy', self.cb_y_compress_noisy, 10)
        self.sub_x_compress_tilt_noisy = self.create_subscription(
            Float32, '/gaussian_field/x_compress_tilt/temperature_noisy', self.cb_x_compress_tilt_noisy, 10)
        self.sub_y_compress_tilt_noisy = self.create_subscription(
            Float32, '/gaussian_field/y_compress_tilt/temperature_noisy', self.cb_y_compress_tilt_noisy, 10)

        self.cur = None
        self.goal = None
        self.sampled = False
        self.xhat = None  # EKF estimated x position
        self.yhat = None  # EKF estimated y position

        # Temperature storage for all fields
        self.temp_radial = None
        self.temp_x_compress = None
        self.temp_y_compress = None
        self.temp_x_compress_tilt = None
        self.temp_y_compress_tilt = None
        self.temp_radial_noisy = None
        self.temp_x_compress_noisy = None
        self.temp_y_compress_noisy = None
        self.temp_x_compress_tilt_noisy = None
        self.temp_y_compress_tilt_noisy = None

        # Get the package directory and use existing data folder
        try:
            pkg_dir = get_package_share_directory('sampling')
            # Go up from install/share to src
            workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(pkg_dir))))
            data_dir = os.path.join(workspace_root, 'src', 'sampling', 'data', 'samples')
        except:
            # Fallback to current directory if package not found
            data_dir = os.path.join(os.getcwd(), 'src', 'sampling', 'data', 'samples')

        # Add timestamp to filename
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")

        # Create two separate CSV files - one for clean, one for noisy
        csv_path_clean = os.path.join(data_dir, f'samples_clean_{timestamp_str}.csv')
        csv_path_noisy = os.path.join(data_dir, f'samples_noisy_{timestamp_str}.csv')

        self.csv_clean = open(csv_path_clean, 'w', newline='')
        self.writer_clean = csv.writer(self.csv_clean)
        self.writer_clean.writerow([
            'x_c', 'y_c', 'xhat', 'yhat', 'timestamp',
            'radial', 'x_compress', 'y_compress', 'x_compress_tilt', 'y_compress_tilt'
        ])

        self.csv_noisy = open(csv_path_noisy, 'w', newline='')
        self.writer_noisy = csv.writer(self.csv_noisy)
        self.writer_noisy.writerow([
            'x_c', 'y_c', 'xhat', 'yhat', 'timestamp',
            'radial_noisy', 'x_compress_noisy', 'y_compress_noisy',
            'x_compress_tilt_noisy', 'y_compress_tilt_noisy'
        ])

        self.get_logger().info(f"Saving clean samples to {csv_path_clean}")
        self.get_logger().info(f"Saving noisy samples to {csv_path_noisy}")

        self.timer = self.create_timer(0.1, self.loop)

    def cb_goal(self, msg):
        self.goal = np.array([msg.point.x, msg.point.y])
        self.sampled = False
        self.get_logger().info(f"New goal set: ({msg.point.x:.1f}, {msg.point.y:.1f})")

    def cb_odom(self, msg):
        # VehicleOdometry contains EKF estimate (xhat, yhat)
        # PX4 uses NED frame, convert to ENU for ROS
        p_ned = np.array([msg.position[0], msg.position[1], msg.position[2]], dtype=float)
        p_enu = ned_to_enu(p_ned)
        self.xhat = float(p_enu[0])  # East
        self.yhat = float(p_enu[1])  # North
        self.cur = np.array([self.xhat, self.yhat])

    # Callbacks for clean temperatures
    def cb_radial(self, msg):
        self.temp_radial = msg.data

    def cb_x_compress(self, msg):
        self.temp_x_compress = msg.data

    def cb_y_compress(self, msg):
        self.temp_y_compress = msg.data

    def cb_x_compress_tilt(self, msg):
        self.temp_x_compress_tilt = msg.data

    def cb_y_compress_tilt(self, msg):
        self.temp_y_compress_tilt = msg.data

    # Callbacks for noisy temperatures
    def cb_radial_noisy(self, msg):
        self.temp_radial_noisy = msg.data

    def cb_x_compress_noisy(self, msg):
        self.temp_x_compress_noisy = msg.data

    def cb_y_compress_noisy(self, msg):
        self.temp_y_compress_noisy = msg.data

    def cb_x_compress_tilt_noisy(self, msg):
        self.temp_x_compress_tilt_noisy = msg.data

    def cb_y_compress_tilt_noisy(self, msg):
        self.temp_y_compress_tilt_noisy = msg.data

    def loop(self):
        if self.cur is None or self.goal is None:
            return
        if self.sampled:
            return

        # Check if all temperature readings are available
        if any(t is None for t in [
            self.temp_radial, self.temp_x_compress, self.temp_y_compress,
            self.temp_x_compress_tilt, self.temp_y_compress_tilt,
            self.temp_radial_noisy, self.temp_x_compress_noisy, self.temp_y_compress_noisy,
            self.temp_x_compress_tilt_noisy, self.temp_y_compress_tilt_noisy
        ]):
            return

        dist = np.linalg.norm(self.cur - self.goal)
        if dist < NEAR_RADIUS:
            timestamp = time.time()
            xhat = float(self.xhat) if self.xhat is not None else float('nan')
            yhat = float(self.yhat) if self.yhat is not None else float('nan')

            # Write clean data
            self.writer_clean.writerow([
                self.goal[0],
                self.goal[1],
                xhat,
                yhat,
                timestamp,
                float(self.temp_radial),
                float(self.temp_x_compress),
                float(self.temp_y_compress),
                float(self.temp_x_compress_tilt),
                float(self.temp_y_compress_tilt),
            ])

            # Write noisy data
            self.writer_noisy.writerow([
                self.goal[0],
                self.goal[1],
                xhat,
                yhat,
                timestamp,
                float(self.temp_radial_noisy),
                float(self.temp_x_compress_noisy),
                float(self.temp_y_compress_noisy),
                float(self.temp_x_compress_tilt_noisy),
                float(self.temp_y_compress_tilt_noisy),
            ])

            self.csv_clean.flush()
            self.csv_noisy.flush()
            self.get_logger().info(
                f"✓ Sampled at ({self.goal[0]:.1f}, {self.goal[1]:.1f}), EKF: ({xhat:.2f}, {yhat:.2f}), dist={dist:.2f}m"
            )
            self.sampled = True

    def destroy_node(self):
        self.csv_clean.close()
        self.csv_noisy.close()
        super().destroy_node()

def main():
    rclpy.init()
    node = SampleLogger()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
