#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class Limits:
    ax: float   # m/s^2
    ay: float   # m/s^2
    aw: float   # rad/s^2


class CmdVelSmoother(Node):
    """
    Subscribe  : /cmd_vel_raw (Twist)  (teleop step input)
    Publish    : /cmd_vel     (Twist)  (smoothed / rate-limited)
    Behavior   : acceleration (slew-rate) limit on vx, vy, wz
    Optional   : command timeout -> ramp to zero
    """

    def __init__(self) -> None:
        super().__init__("cmd_vel_smoother")

        # 能用launch/命令行来修改Parameters
        self.declare_parameter("input_topic", "/cmd_vel_raw")
        self.declare_parameter("output_topic", "/cmd_vel")
        self.declare_parameter("rate_hz", 100.0)
        self.declare_parameter("ax_max", 1.0)    # m/s^2
        self.declare_parameter("ay_max", 1.0)    # m/s^2
        self.declare_parameter("aw_max", 2.0)    # rad/s^2
        self.declare_parameter("timeout_s", 0.25)

        in_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        out_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.rate_hz = float(self.get_parameter("rate_hz").value)

        self.lim = Limits(
            ax=float(self.get_parameter("ax_max").value),
            ay=float(self.get_parameter("ay_max").value),
            aw=float(self.get_parameter("aw_max").value),
        )
        self.timeout_s = float(self.get_parameter("timeout_s").value)

        self.sub = self.create_subscription(Twist, in_topic, self.on_cmd, 10)
        self.pub = self.create_publisher(Twist, out_topic, 10)

        # State
        self.cmd = Twist()      # latest desired command
        self.out = Twist()      # current output command
        self.last_cmd_time = self.get_clock().now()

        dt = 1.0 / max(1.0, self.rate_hz)
        self.timer = self.create_timer(dt, self.on_timer)

        self.get_logger().info(
            f"cmd_vel_smoother: {in_topic} -> {out_topic}, rate={self.rate_hz}Hz, "
            f"ax={self.lim.ax}, ay={self.lim.ay}, aw={self.lim.aw}, timeout={self.timeout_s}s"
        )

    def on_cmd(self, msg: Twist) -> None:
        self.cmd = msg
        self.last_cmd_time = self.get_clock().now()

    def on_timer(self) -> None:
        now = self.get_clock().now()
        dt = 1.0 / max(1.0, self.rate_hz)

        # If no new command for a while, ramp target to zero (safer)
        if (now - self.last_cmd_time).nanoseconds * 1e-9 > self.timeout_s:
            target = Twist()
        else:
            target = self.cmd

        # Rate-limit each component
        self.out.linear.x = self._slew(self.out.linear.x, target.linear.x, self.lim.ax, dt)
        self.out.linear.y = self._slew(self.out.linear.y, target.linear.y, self.lim.ay, dt)
        self.out.angular.z = self._slew(self.out.angular.z, target.angular.z, self.lim.aw, dt)

        # Keep other components zero
        self.out.linear.z = 0.0
        self.out.angular.x = 0.0
        self.out.angular.y = 0.0

        self.pub.publish(self.out)

    @staticmethod
    def _slew(v_out: float, v_cmd: float, a_max: float, dt: float) -> float:
        if a_max <= 0.0 or dt <= 0.0:
            return v_cmd
        dv_max = a_max * dt
        dv = clamp(v_cmd - v_out, -dv_max, dv_max)
        v_new = v_out + dv

        # avoid -0.0 noise
        if abs(v_new) < 1e-12:
            v_new = 0.0
        return v_new


def main() -> None:
    rclpy.init()
    node = CmdVelSmoother()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
