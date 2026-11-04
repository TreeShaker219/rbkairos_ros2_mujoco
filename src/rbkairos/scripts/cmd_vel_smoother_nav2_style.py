#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仿照 nav2_velocity_smoother 的加速度/减速度限幅逻辑：
- 周期性以 smoothing_frequency 限斜率：加速用 max_accel，减速用 max_decel（应为负数）。
- 可选整体缩放（scale_velocities）保持方向，避免单轴超限。
- 超时后自动减速到 0。

默认只处理平面 3 自由度：vx, vy, wz。
"""
from __future__ import annotations

import math
from typing import List

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class CmdVelSmootherNav2Style(Node):
    def __init__(self) -> None:
        super().__init__("cmd_vel_smoother_nav2_style")

        self.input_topic = self.declare_parameter("input_topic", "/cmd_vel").get_parameter_value().string_value
        self.output_topic = self.declare_parameter("output_topic", "/cmd_vel_smoothed").get_parameter_value().string_value
        self.smoothing_frequency = float(self.declare_parameter("smoothing_frequency", 400.0).value)
        self.velocity_timeout = float(self.declare_parameter("velocity_timeout", 1.0).value)
        # scale_velocities=False: 逐轴独立限幅（默认更直观）
        # scale_velocities=True : 整体矢量缩放（Nav2 思路，保持方向）
        self.scale_velocities = bool(self.declare_parameter("scale_velocities", False).value)
        # max/min velocity
        # 允许 mecanum 侧移，默认 vx/vy 同限
        # 默认上限 1 m/s（vx/vy），偏航保持 2.5 rad/s
        self.max_vel = self._get_vector_param("max_velocity", [1.0, 1.0, 2.5])
        self.min_vel = self._get_vector_param("min_velocity", [-1.0, -1.0, -2.5])
        # 加速度应为正，减速度应为负
        #self.max_accel = self._get_vector_param("max_accel", [2.5, 2.5, 3.2])
        #self.max_decel = self._get_vector_param("max_decel", [-2.5, -2.5, -3.2])
        self.max_accel = self._get_vector_param("max_accel", [1000.0, 1000.0, 1000.0])
        self.max_decel = self._get_vector_param("max_decel", [-1000.0, -1000.0, -1000.0])
        self.deadband = self._get_vector_param("deadband_velocity", [0.0, 0.0, 0.0])

        self._validate_params()

        self.cmd = Twist()   # 最新目标
        self.out = Twist()   # 当前输出
        self.received_first = False
        self.last_cmd_time = self.get_clock().now()

        self.pub = self.create_publisher(Twist, self.output_topic, 10)
        self.sub = self.create_subscription(Twist, self.input_topic, self._cb, 10)

        period = 1.0 / self.smoothing_frequency
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            f"nav2-style smoother: {self.input_topic} -> {self.output_topic}, "
            f"freq={self.smoothing_frequency} Hz, scale_velocities={self.scale_velocities}")

    def _get_vector_param(self, name: str, default: List[float]) -> List[float]:
        val = self.declare_parameter(name, default).get_parameter_value().double_array_value
        vec = list(val) if val else list(default)
        if len(vec) != 3:
            self.get_logger().warn(f"{name} 长度应为3，收到 {vec}，回退默认 {default}")
            vec = list(default)
        return vec

    def _validate_params(self) -> None:
        # 频率必须大于 0
        if self.smoothing_frequency <= 0.0:
            self.get_logger().error(f"smoothing_frequency 必须 > 0，当前 {self.smoothing_frequency}")
            raise RuntimeError("Invalid smoothing_frequency")
        # 加速度为正，减速度为负，速度上下限需一致
        for i, (a, d, vmin, vmax) in enumerate(zip(self.max_accel, self.max_decel, self.min_vel, self.max_vel)):
            if a < 0.0:
                self.get_logger().error(f"max_accel[{i}] 必须 >= 0，当前 {a}")
                raise RuntimeError("Invalid max_accel")
            if d > 0.0:
                self.get_logger().error(f"max_decel[{i}] 必须 <= 0，当前 {d}")
                raise RuntimeError("Invalid max_decel")
            if vmin > vmax:
                self.get_logger().error(f"min_velocity[{i}] 不应大于 max_velocity[{i}]，当前 {vmin} > {vmax}")
                raise RuntimeError("Invalid velocity limits")

    def _cb(self, msg: Twist) -> None:
        self.cmd = msg
        self.last_cmd_time = self.get_clock().now()
        self.received_first = True

    def _find_eta(self, curr: float, target: float, accel: float, decel: float, dt: float) -> float:
        """严格对齐 nav2 findEtaConstraint：用与 _apply_constraints 相同的加/减速判据"""
        dv = target - curr
        if abs(dv) < 1e-12:
            return -1.0

        if abs(target) >= abs(curr) and curr * target >= 0.0:
            # 加速分支
            dv_max = accel * dt
            dv_min = -accel * dt
        else:
            # 减速分支（decel 为负）
            dv_max = -decel * dt
            dv_min = decel * dt

        if dv > dv_max:
            return dv_max / dv
        if dv < dv_min:
            return dv_min / dv
        return -1.0

    def _apply_constraints(self, curr: float, target: float, accel: float, decel: float, eta: float, dt: float) -> float:
        dv = target - curr
        # 加速或减速选择对应斜率
        if abs(target) >= abs(curr) and curr * target >= 0.0:
            dv_max = accel * dt
            dv_min = -accel * dt
        else:
            dv_max = -decel * dt  # decel 为负
            dv_min = decel * dt
        return curr + clamp((eta if eta > 0 else 1.0) * dv, dv_min, dv_max)

    def _tick(self) -> None:
        if not self.received_first:
            return

        now = self.get_clock().now()
        dt = 1.0 / self.smoothing_frequency

        # 超时则目标清零
        if (now - self.last_cmd_time).nanoseconds * 1e-9 > self.velocity_timeout:
            self.cmd = Twist()

        # 裁剪速度上限
        cmd = Twist()
        cmd.linear.x = clamp(self.cmd.linear.x, self.min_vel[0], self.max_vel[0])
        cmd.linear.y = clamp(self.cmd.linear.y, self.min_vel[1], self.max_vel[1])
        cmd.angular.z = clamp(self.cmd.angular.z, self.min_vel[2], self.max_vel[2])

        # 计算缩放系数 eta（如需要整矢量缩放）
        eta = 1.0
        if self.scale_velocities:
            for (c, t, a, d) in [
                (self.out.linear.x, cmd.linear.x, self.max_accel[0], self.max_decel[0]),
                (self.out.linear.y, cmd.linear.y, self.max_accel[1], self.max_decel[1]),
                (self.out.angular.z, cmd.angular.z, self.max_accel[2], self.max_decel[2]),
            ]:
                e = self._find_eta(c, t, a, d, dt)
                if 0.0 < e < eta:
                    eta = e

        # 应用斜率限制
        self.out.linear.x = self._apply_constraints(self.out.linear.x, cmd.linear.x, self.max_accel[0], self.max_decel[0], eta, dt)
        self.out.linear.y = self._apply_constraints(self.out.linear.y, cmd.linear.y, self.max_accel[1], self.max_decel[1], eta, dt)
        self.out.angular.z = self._apply_constraints(self.out.angular.z, cmd.angular.z, self.max_accel[2], self.max_decel[2], eta, dt)

        # deadband
        if abs(self.out.linear.x) < self.deadband[0]:
            self.out.linear.x = 0.0
        if abs(self.out.linear.y) < self.deadband[1]:
            self.out.linear.y = 0.0
        if abs(self.out.angular.z) < self.deadband[2]:
            self.out.angular.z = 0.0

        self.pub.publish(self.out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelSmootherNav2Style()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
