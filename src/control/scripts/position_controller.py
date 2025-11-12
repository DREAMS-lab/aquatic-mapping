#!/usr/bin/env python3
import time, threading
from collections import deque
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry
from geometry_msgs.msg import PointStamped

NEAR_RADIUS = 2.0
STABLE_SECS = 3.0
STD_THRESH  = 0.05
WIN_SIZE    = 50  # 50*0.05s ≈ 2.5s window

class BaselineController(Node):
    def __init__(self):
        super().__init__('baseline_controller')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.pub_off = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.pub_sp  = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.pub_cmd = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)
        self.pub_tgt = self.create_publisher(PointStamped, '/circle/target_position', 10)

        self.sub_odom = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.cb_odom, qos)

        self.cur  = np.zeros(3)
        self.goal = np.zeros(3)
        self.have_odom = False

        self.goal_active = False
        self.near_since = None
        self.hist = deque(maxlen=WIN_SIZE)

        self.counter = 0
        self.timer = self.create_timer(0.05, self.loop)

    def cb_odom(self, msg: VehicleOdometry):
        self.cur[:] = [msg.position[0], msg.position[1], msg.position[2]]
        self.have_odom = True
        if not self.goal_active:
            return

        dist = float(np.linalg.norm(self.cur[:2] - self.goal[:2]))
        if dist < NEAR_RADIUS:
            if self.near_since is None:
                self.near_since = time.time()
            self.hist.append(self.cur[:2].copy())
            if len(self.hist) == self.hist.maxlen and (time.time() - self.near_since) >= STABLE_SECS:
                std_xy = np.std(np.array(self.hist), axis=0)
                if max(std_xy) < STD_THRESH:
                    # final error measured ONLY to the typed goal (no hacks)
                    print(f"FINAL_ERROR_BASELINE {dist:.3f}")
                    # publish typed goal so your DataLogger writes a row
                    tgt = PointStamped()
                    tgt.header.stamp = self.get_clock().now().to_msg()
                    tgt.header.frame_id = 'map'
                    tgt.point.x, tgt.point.y, tgt.point.z = float(self.goal[0]), float(self.goal[1]), 0.0
                    self.pub_tgt.publish(tgt)

                    self.goal_active = False
                    self.near_since = None
                    self.hist.clear()
        else:
            self.near_since = None
            self.hist.clear()

    def loop(self):
        o = OffboardControlMode()
        o.position = True
        o.velocity = o.acceleration = o.attitude = o.body_rate = False
        o.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_off.publish(o)

        if self.counter == 10:
            self._set_mode(); self._arm()
        self.counter += 1

        sp = TrajectorySetpoint()
        sp.position = [float(self.goal[0]), float(self.goal[1]), 0.0]
        sp.velocity = [float('nan')]*3
        sp.yaw = float('nan')
        sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_sp.publish(sp)

    def go_to(self, x, y):
        self.goal[:] = [x, y, 0.0]
        self.goal_active = True
        self.near_since = None
        self.hist.clear()
        print(f"SEND_BASELINE {x:.2f} {y:.2f}")

    # MAVLink helpers
    def _cmd(self, c, p1=0.0, p2=0.0):
        m = VehicleCommand()
        m.param1, m.param2 = float(p1), float(p2)
        m.command = c
        m.target_system = 1
        m.target_component = 1
        m.source_system = 1
        m.source_component = 1
        m.from_external = True
        m.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_cmd.publish(m)
    def _arm(self):      self._cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
    def _disarm(self):   self._cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
    def _set_mode(self): self._cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

def main(args=None):
    rclpy.init(args=args)
    node = BaselineController()
    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True); th.start()
    time.sleep(0.5)
    try:
        while rclpy.ok():
            s = input("Enter X Y (q to quit): ").strip()
            if s.lower() in ('q','quit','exit'): break
            try:
                x, y = map(float, s.split())
            except Exception:
                print("bad input"); continue
            node.go_to(x, y)
    except KeyboardInterrupt:
        pass
    finally:
        try: node._disarm()
        except: pass
        try: node.destroy_node()
        finally:
            if rclpy.ok(): rclpy.shutdown()
        th.join(timeout=1.0)

if __name__ == "__main__":
    main()
