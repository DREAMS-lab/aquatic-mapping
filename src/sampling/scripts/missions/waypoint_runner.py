#!/usr/bin/env python3
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import PointStamped
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry

# PX4 VehicleOdometry is NED; our experiment grid/field uses ENU (world)
def ned_to_enu(p_xyz: np.ndarray) -> np.ndarray:
    """PX4 -> ROS: (x_north, y_east, z_down) -> (x_east, y_north, z_up)"""
    x_n, y_e, z_d = float(p_xyz[0]), float(p_xyz[1]), float(p_xyz[2])
    return np.array([y_e, x_n, -z_d], dtype=float)

def enu_to_ned(p_xyz: np.ndarray) -> np.ndarray:
    """ROS -> PX4: (x_east, y_north, z_up) -> (x_north, y_east, z_down)"""
    x_e, y_n, z_u = float(p_xyz[0]), float(p_xyz[1]), float(p_xyz[2])
    return np.array([y_n, x_e, -z_u], dtype=float)

# Grid pattern covering the field from (1,1) to (24,24)
# Lawn-mower pattern with spacing of ~5.75 units (5 rows x 5 cols)
WAYPOINTS = [
    # Row 1: y=1, sweep left to right
    (1.0, 1.0), (6.75, 1.0), (12.5, 1.0), (18.25, 1.0), (24.0, 1.0),
    # Row 2: y=6.75, sweep right to left
    (24.0, 6.75), (18.25, 6.75), (12.5, 6.75), (6.75, 6.75), (1.0, 6.75),
    # Row 3: y=12.5, sweep left to right
    (1.0, 12.5), (6.75, 12.5), (12.5, 12.5), (18.25, 12.5), (24.0, 12.5),
    # Row 4: y=18.25, sweep right to left
    (24.0, 18.25), (18.25, 18.25), (12.5, 18.25), (6.75, 18.25), (1.0, 18.25),
    # Row 5: y=24, sweep left to right
    (1.0, 24.0), (6.75, 24.0), (12.5, 24.0), (18.25, 24.0), (24.0, 24.0),
]

WAYPOINT_RADIUS = 1.5  # meters - consider waypoint reached within this distance

class WaypointRunner(Node):
    def __init__(self):
        super().__init__('waypoint_runner')

        self.pub_off = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', 1)
        self.pub_sp = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 1)
        self.pub_cmd = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', 1)

        self.pub_goal = self.create_publisher(
            PointStamped, '/sampling/commanded_location', 10)
        
        # Subscribe to odometry to track progress
        qos_px4 = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.sub_odom = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos_px4)

        self.idx = 0
        self.timer = self.create_timer(0.05, self.loop)
        self.sent = False
        self.counter = 0
        self.current_pos = None
        self.waypoint_reached_time = None
        self.hold_duration = 3.0  # seconds to hold at waypoint before moving to next
        
        self.get_logger().info('===== Waypoint Runner Started =====')
        self.get_logger().info(f'Total waypoints: {len(WAYPOINTS)}')
        self.get_logger().info('NOTE: Make sure PX4 simulation is running!')
        self.get_logger().info('  Start with: MicroXRCEAgent udp4 -p 8888')
        self.get_logger().info('  Or use: ros2 launch px4_offboard offboard.launch.py')

    def odom_callback(self, msg):
        """Track current position from odometry"""
        p_ned = np.array([msg.position[0], msg.position[1], msg.position[2]], dtype=float)
        p_enu = ned_to_enu(p_ned)
        self.current_pos = np.array([p_enu[0], p_enu[1]], dtype=float)
    
    def loop(self):
        off = OffboardControlMode()
        off.position = True
        off.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_off.publish(off)

        if self.counter == 10:
            self._set_mode()
            self._arm()
        self.counter += 1

        if self.idx >= len(WAYPOINTS):
            self.get_logger().info("All waypoints completed!", throttle_duration_sec=5.0)
            return

        x, y = WAYPOINTS[self.idx]

        sp = TrajectorySetpoint()
        # PX4 expects NED setpoints; convert ENU -> NED
        p_ned = enu_to_ned(np.array([x, y, 0.0], dtype=float))
        sp.position = [float(p_ned[0]), float(p_ned[1]), float(p_ned[2])]
        sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_sp.publish(sp)

        if not self.sent:
            msg = PointStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'world'
            msg.point.x = x
            msg.point.y = y
            self.pub_goal.publish(msg)

            self.get_logger().info(f"→ Waypoint {self.idx + 1}/{len(WAYPOINTS)}: ({x:.1f}, {y:.1f})")
            self.sent = True
        
        # Check if we've reached the waypoint
        if self.current_pos is not None:
            goal = np.array([x, y])
            dist = np.linalg.norm(self.current_pos - goal)
            
            if dist < WAYPOINT_RADIUS:
                if self.waypoint_reached_time is None:
                    self.waypoint_reached_time = time.time()
                    self.get_logger().info(f"✓ Reached waypoint {self.idx + 1} (dist: {dist:.2f}m)")
                
                # Hold at waypoint for a few seconds before moving on
                if time.time() - self.waypoint_reached_time > self.hold_duration:
                    self.next_waypoint()
            else:
                # Reset timer if we drift away
                self.waypoint_reached_time = None

    def next_waypoint(self):
        """Move to the next waypoint"""
        self.idx += 1
        self.sent = False
        self.waypoint_reached_time = None
        if self.idx < len(WAYPOINTS):
            self.get_logger().info(f"Moving to next waypoint...")

    def _cmd(self, c, p1=0.0, p2=0.0):
        m = VehicleCommand()
        m.command = c
        m.param1 = float(p1)
        m.param2 = float(p2)
        m.target_system = 1
        m.target_component = 1
        m.source_system = 1
        m.source_component = 1
        m.from_external = True
        m.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_cmd.publish(m)

    def _arm(self):
        self._cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

    def _set_mode(self):
        self._cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

def main():
    rclpy.init()
    node = WaypointRunner()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
