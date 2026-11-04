#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
订阅 /mujoco_dof_state，把启动后的 15 秒内的 MuJoCo DOF 位置、速度、
加速度、约束力、控制器施加力和约束分解广义力写入 CSV。

特点：
- 从节点启动时开始计时（墙钟 15 秒），不中断正在运行的仿真。
- 以 /mujoco_dof_state 的 header 时间戳为记录时间，按每条消息写若干行。
- 加速度来自 MujocoDofState.qacc
- 约束力来自 MujocoDofState.qfrc_constraint
- 施加力来自 MujocoDofState.qfrc_applied

输出：
- 默认写到 data/mujoco_dof_states_YYYYmmdd_HHMMSS.csv
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional
import math

import rclpy
from mujoco_ros2_control_msgs.msg import MujocoDofState
from rclpy.node import Node


DOF_TOPIC = "/mujoco_dof_state"


class JointStateRecorder(Node):
    def __init__(self, duration_sec: float = 15.0) -> None:
        super().__init__("joint_state_recorder")
        self.duration_sec = float(duration_sec)
        self.start_wall = self.get_clock().now()

        out_dir = Path("data")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_path = out_dir / f"mujoco_dof_states_{timestamp}.csv"

        self.file = self.out_path.open("w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            [
                "time_sec",
                "dof_name",
                "position",
                "velocity",
                "acceleration",
                "constraint",
                "applied",
                "actuator",
                "passive",
                "bias",
                "constraint_equality",
                "constraint_friction",
                "constraint_limit",
                "constraint_contact",
                "constraint_other",
                "nefc",
                "ncon",
            ]
        )

        # 过滤不需要的关节前缀（如 freejoint 生成的 root#0-5）
        self.ignore_prefixes = ("root#",)

        self.sub_dof = self.create_subscription(
            MujocoDofState, DOF_TOPIC, self._cb_dof, 50
        )

        # 定期检查是否到时
        self.timer = self.create_timer(0.5, self._check_stop)

        self.get_logger().info(
            f"开始记录 {DOF_TOPIC}，持续 {self.duration_sec} s，输出 {self.out_path}"
        )

    def _cb_dof(self, msg: MujocoDofState) -> None:
        # 以 /mujoco_dof_state 的时间戳为准，所有 DOF 数组均为 MuJoCo nv 顺序。
        time_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        t = time_ns * 1e-9

        names = list(msg.dof_names)
        positions = list(msg.qpos)
        velocities = list(msg.qvel)
        accelerations = list(msg.qacc)
        constraint = list(msg.qfrc_constraint)
        applied = list(msg.qfrc_applied)
        actuator = list(msg.qfrc_actuator)
        passive = list(msg.qfrc_passive)
        bias = list(msg.qfrc_bias)
        constraint_equality = list(msg.qfrc_constraint_equality)
        constraint_friction = list(msg.qfrc_constraint_friction)
        constraint_limit = list(msg.qfrc_constraint_limit)
        constraint_contact = list(msg.qfrc_constraint_contact)
        constraint_other = list(msg.qfrc_constraint_other)

        for i, name in enumerate(names):
            if name.startswith(self.ignore_prefixes):
                continue
            pos = positions[i] if i < len(positions) else 0.0
            vel = velocities[i] if i < len(velocities) else 0.0
            acc = accelerations[i] if i < len(accelerations) else 0.0
            cst = constraint[i] if i < len(constraint) else math.nan
            app = applied[i] if i < len(applied) else math.nan
            act = actuator[i] if i < len(actuator) else math.nan
            pas = passive[i] if i < len(passive) else math.nan
            bia = bias[i] if i < len(bias) else math.nan
            cst_eq = constraint_equality[i] if i < len(constraint_equality) else math.nan
            cst_fric = constraint_friction[i] if i < len(constraint_friction) else math.nan
            cst_lim = constraint_limit[i] if i < len(constraint_limit) else math.nan
            cst_con = constraint_contact[i] if i < len(constraint_contact) else math.nan
            cst_other = constraint_other[i] if i < len(constraint_other) else math.nan
            self.writer.writerow(
                [
                    f"{t:.9f}",
                    name,
                    pos,
                    vel,
                    acc,
                    cst,
                    app,
                    act,
                    pas,
                    bia,
                    cst_eq,
                    cst_fric,
                    cst_lim,
                    cst_con,
                    cst_other,
                    int(msg.nefc),
                    int(msg.ncon),
                ]
            )

    def _check_stop(self) -> None:
        elapsed = (self.get_clock().now() - self.start_wall).nanoseconds * 1e-9
        if elapsed >= self.duration_sec:
            self.get_logger().info(f"录制结束，耗时 {elapsed:.2f} s，文件保存在 {self.out_path}")
            self.file.close()
            rclpy.shutdown()

    def destroy_node(self) -> bool:
        if not self.file.closed:
            self.file.close()
        return super().destroy_node()


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = JointStateRecorder(duration_sec=15.0)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
