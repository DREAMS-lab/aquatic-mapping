#!/usr/bin/env python3
import time
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand

class GotoCommander(Node):
    def __init__(self):
        super().__init__('goto_commander')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.pub_off = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.pub_sp  = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.pub_cmd = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)

        self.goal_x = 0.0
        self.goal_y = 0.0

        self.counter = 0
        self.timer = self.create_timer(0.05, self.loop)  # 20 Hz setpoint stream

    def loop(self):
        # Offboard keep-alive
        off = OffboardControlMode()
        off.position = True
        off.velocity = off.acceleration = off.attitude = off.body_rate = False
        off.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_off.publish(off)

        # Switch to offboard + arm once after a short stream
        if self.counter == 10:
            self._set_mode_offboard()
            self._arm()
        self.counter += 1

        # Position setpoint
        sp = TrajectorySetpoint()
        sp.position = [float(self.goal_x), float(self.goal_y), 0.0]
        sp.velocity = [float('nan')] * 3
        sp.yaw = float('nan')
        sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_sp.publish(sp)

    def set_goal(self, x_c: float, y_c: float):
        self.goal_x = float(x_c)
        self.goal_y = float(y_c)
        self.get_logger().info(f"Commanded goal x_c=({self.goal_x:.2f}, {self.goal_y:.2f})")

    # PX4 command helpers
    def _cmd(self, command: int, p1: float = 0.0, p2: float = 0.0):
        m = VehicleCommand()
        m.param1 = float(p1)
        m.param2 = float(p2)
        m.command = int(command)
        m.target_system = 1
        m.target_component = 1
        m.source_system = 1
        m.source_component = 1
        m.from_external = True
        m.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_cmd.publish(m)

    def _arm(self):
        self._cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

    def _disarm(self):
        self._cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)

    def _set_mode_offboard(self):
        # param1=1 (custom mode), param2=6 (PX4 offboard)
        self._cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

def main(args=None):
    rclpy.init(args=args)
    node = GotoCommander()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    time.sleep(0.5)
    try:
        while rclpy.ok():
            s = input("Enter X Y (q to quit): ").strip()
            if s.lower() in ('q', 'quit', 'exit'):
                break
            try:
                x, y = map(float, s.split())
            except Exception:
                print("bad input")
                continue
            node.set_goal(x, y)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._disarm()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == "__main__":
    main()
