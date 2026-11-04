#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drive the mecanum base forward/backward with a smooth open-loop profile and
record MuJoCo joint data, commanded velocity, and base ground-truth odometry.

Expected setup:
  ros2 launch rbkairos mujoco_mecanum_drive.launch.py

Example:
  python3 src/rbkairos/scripts/back_and_forth_motion_recorder.py
  python3 src/rbkairos/scripts/back_and_forth_motion_recorder.py --distance 0.3 --speed 0.15 --cycles 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist
from mujoco_ros2_control_msgs.msg import MujocoDofState
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


DOF_TOPIC_DEFAULT = "/mujoco_dof_state"


@dataclass(frozen=True)
class MotionProfile:
    distance: float
    requested_speed: float
    accel: float
    peak_speed: float
    accel_time: float
    cruise_time: float
    duration: float

    @classmethod
    def from_limits(cls, distance: float, speed: float, accel: float) -> "MotionProfile":
        accel_time_to_speed = speed / accel
        accel_distance = 0.5 * accel * accel_time_to_speed * accel_time_to_speed

        if 2.0 * accel_distance >= distance:
            peak_speed = math.sqrt(distance * accel)
            accel_time = peak_speed / accel
            cruise_time = 0.0
        else:
            peak_speed = speed
            accel_time = accel_time_to_speed
            cruise_distance = distance - 2.0 * accel_distance
            cruise_time = cruise_distance / peak_speed

        return cls(
            distance=distance,
            requested_speed=speed,
            accel=accel,
            peak_speed=peak_speed,
            accel_time=accel_time,
            cruise_time=cruise_time,
            duration=2.0 * accel_time + cruise_time,
        )

    def velocity_at(self, elapsed: float, direction: int) -> float:
        if elapsed < 0.0 or elapsed >= self.duration:
            return 0.0

        if elapsed < self.accel_time:
            velocity = self.accel * elapsed
        elif elapsed < self.accel_time + self.cruise_time:
            velocity = self.peak_speed
        else:
            decel_elapsed = elapsed - self.accel_time - self.cruise_time
            velocity = max(0.0, self.peak_speed - self.accel * decel_elapsed)

        return float(direction) * velocity


@dataclass(frozen=True)
class MotionSegment:
    phase: str
    cycle: int
    direction: int
    duration: float


def stamp_to_sec(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def build_segments(cycles: int, profile: MotionProfile, pause: float) -> list[MotionSegment]:
    segments: list[MotionSegment] = []
    for cycle in range(1, cycles + 1):
        segments.append(MotionSegment(f"cycle_{cycle}_forward", cycle, 1, profile.duration))
        if pause > 0.0:
            segments.append(MotionSegment(f"cycle_{cycle}_pause_after_forward", cycle, 0, pause))
        segments.append(MotionSegment(f"cycle_{cycle}_backward", cycle, -1, profile.duration))
        if pause > 0.0:
            segments.append(MotionSegment(f"cycle_{cycle}_pause_after_backward", cycle, 0, pause))
    return segments


class BackAndForthMotionRecorder(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("back_and_forth_motion_recorder")

        self.args = args
        self.cmd_topic = args.cmd_topic
        self.dof_topic = args.dof_topic
        self.odom_topic = args.odom_topic
        self.startup_timeout = float(args.startup_timeout)
        self.control_rate = float(args.rate)
        self.node_start_wall = time.monotonic()
        self.wait_start_wall = self.node_start_wall
        self.motion_start_wall: Optional[float] = None
        self.started = False
        self.finished = False
        self.startup_warning_logged = False
        self.ignore_prefixes = ("root#",)

        self.profile = MotionProfile.from_limits(args.distance, args.speed, args.accel)
        self.segments = build_segments(args.cycles, self.profile, args.pause)
        self.segment_index = 0
        self.segment_start_elapsed = 0.0

        created_at = datetime.now()
        timestamp = created_at.strftime("%Y%m%d_%H%M%S")
        self.created_at = created_at.isoformat(timespec="seconds")
        self.motion_started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.output_dir = Path(args.data_dir) / f"back_and_forth_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.joint_path = self.output_dir / "joint_states.csv"
        self.cmd_path = self.output_dir / "cmd_vel.csv"
        self.odom_path = self.output_dir / "odom.csv"
        self.metadata_path = self.output_dir / "metadata.json"

        self.joint_file = self.joint_path.open("w", newline="")
        self.cmd_file = self.cmd_path.open("w", newline="")
        self.odom_file = self.odom_path.open("w", newline="")
        self.joint_writer = csv.writer(self.joint_file)
        self.cmd_writer = csv.writer(self.cmd_file)
        self.odom_writer = csv.writer(self.odom_file)
        self._write_csv_headers()

        self.latest_cmd: Dict[str, Any] = {
            "phase": "waiting",
            "cycle": 0,
            "direction": 0,
            "vx": 0.0,
            "vy": 0.0,
            "wz": 0.0,
        }
        self.latest_odom: Dict[str, float] = {
            "x": math.nan,
            "y": math.nan,
            "yaw": math.nan,
            "vx": math.nan,
            "vy": math.nan,
            "wz": math.nan,
        }
        self.received_dof = False
        self.received_odom = False

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.dof_sub = self.create_subscription(
            MujocoDofState, self.dof_topic, self._on_mujoco_dof_state, 50
        )
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self._on_odom, qos_profile_sensor_data
        )
        self.timer = self.create_timer(1.0 / self.control_rate, self._on_timer)

        self._write_metadata("initialized")
        self.get_logger().info(
            f"Waiting for topics before motion. Output directory: {self.output_dir}"
        )

    def _write_csv_headers(self) -> None:
        self.joint_writer.writerow(
            [
                "time_sec",
                "joint_name",
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
                "cmd_vx",
                "cmd_vy",
                "cmd_wz",
                "odom_x",
                "odom_y",
                "odom_yaw",
                "odom_vx",
                "odom_vy",
                "odom_wz",
            ]
        )
        self.cmd_writer.writerow(
            [
                "elapsed_sec",
                "ros_time_sec",
                "phase",
                "cycle",
                "direction",
                "cmd_vx",
                "cmd_vy",
                "cmd_wz",
            ]
        )
        self.odom_writer.writerow(
            [
                "time_sec",
                "wall_elapsed_sec",
                "x",
                "y",
                "z",
                "yaw",
                "orientation_x",
                "orientation_y",
                "orientation_z",
                "orientation_w",
                "linear_x",
                "linear_y",
                "linear_z",
                "angular_x",
                "angular_y",
                "angular_z",
            ]
        )

    def _write_metadata(self, status: str) -> None:
        total_duration = sum(segment.duration for segment in self.segments)
        metadata = {
            "status": status,
            "created_at": self.created_at,
            "motion_started_at": self.motion_started_at,
            "finished_at": self.finished_at,
            "node_name": self.get_name(),
            "topics": {
                "cmd_vel": self.cmd_topic,
                "mujoco_dof_state": self.dof_topic,
                "odom": self.odom_topic,
            },
            "parameters": {
                "distance": self.args.distance,
                "speed": self.args.speed,
                "accel": self.args.accel,
                "cycles": self.args.cycles,
                "pause": self.args.pause,
                "rate": self.args.rate,
                "startup_timeout": self.args.startup_timeout,
            },
            "motion_profile": {
                "peak_speed": self.profile.peak_speed,
                "accel_time": self.profile.accel_time,
                "cruise_time": self.profile.cruise_time,
                "leg_duration": self.profile.duration,
                "total_planned_duration": total_duration,
            },
            "output_files": {
                "joint_states": str(self.joint_path),
                "cmd_vel": str(self.cmd_path),
                "odom": str(self.odom_path),
                "metadata": str(self.metadata_path),
            },
        }
        with self.metadata_path.open("w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, indent=2, sort_keys=True)
            file_obj.write("\n")

    def _on_mujoco_dof_state(self, msg: MujocoDofState) -> None:
        self.received_dof = True
        time_sec = stamp_to_sec(msg.header.stamp)

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
            self.joint_writer.writerow(
                [
                    f"{time_sec:.9f}",
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
                    self.latest_cmd["vx"],
                    self.latest_cmd["vy"],
                    self.latest_cmd["wz"],
                    self.latest_odom["x"],
                    self.latest_odom["y"],
                    self.latest_odom["yaw"],
                    self.latest_odom["vx"],
                    self.latest_odom["vy"],
                    self.latest_odom["wz"],
                ]
            )

    def _on_odom(self, msg: Odometry) -> None:
        self.received_odom = True
        pose = msg.pose.pose
        twist = msg.twist.twist
        yaw = yaw_from_quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )

        self.latest_odom = {
            "x": pose.position.x,
            "y": pose.position.y,
            "yaw": yaw,
            "vx": twist.linear.x,
            "vy": twist.linear.y,
            "wz": twist.angular.z,
        }

        self.odom_writer.writerow(
            [
                f"{stamp_to_sec(msg.header.stamp):.9f}",
                f"{time.monotonic() - self.node_start_wall:.9f}",
                pose.position.x,
                pose.position.y,
                pose.position.z,
                yaw,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
                twist.linear.x,
                twist.linear.y,
                twist.linear.z,
                twist.angular.x,
                twist.angular.y,
                twist.angular.z,
            ]
        )

    def _on_timer(self) -> None:
        if self.finished:
            return
        if not self.started:
            if self._ready_to_start():
                self._start_motion()
            else:
                return

        elapsed = time.monotonic() - (self.motion_start_wall or time.monotonic())
        while self.segment_index < len(self.segments):
            segment = self.segments[self.segment_index]
            if elapsed - self.segment_start_elapsed < segment.duration:
                break
            self.segment_start_elapsed += segment.duration
            self.segment_index += 1

        if self.segment_index >= len(self.segments):
            self._finish("completed")
            return

        segment = self.segments[self.segment_index]
        segment_elapsed = elapsed - self.segment_start_elapsed
        vx = 0.0
        if segment.direction != 0:
            vx = self.profile.velocity_at(segment_elapsed, segment.direction)
        self._publish_command(vx, segment.phase, segment.cycle, segment.direction, elapsed)

    def _ready_to_start(self) -> bool:
        missing = []
        if self.cmd_pub.get_subscription_count() == 0:
            missing.append(f"subscriber on {self.cmd_topic}")
        if not self.received_dof:
            missing.append(self.dof_topic)
        if not self.received_odom:
            missing.append(self.odom_topic)

        if not missing:
            return True

        elapsed = time.monotonic() - self.wait_start_wall
        if elapsed >= self.startup_timeout:
            if not self.startup_warning_logged:
                self.get_logger().warn(
                    "Startup timeout reached; starting anyway. Missing: " + ", ".join(missing)
                )
                self.startup_warning_logged = True
            return True

        return False

    def _start_motion(self) -> None:
        self.started = True
        self.motion_start_wall = time.monotonic()
        self.motion_started_at = datetime.now().isoformat(timespec="seconds")
        self.segment_index = 0
        self.segment_start_elapsed = 0.0
        self._write_metadata("running")
        self.get_logger().info(
            f"Starting {self.args.cycles} back-and-forth cycle(s), "
            f"leg_duration={self.profile.duration:.3f}s, peak_speed={self.profile.peak_speed:.3f}m/s"
        )

    def _publish_command(
        self, vx: float, phase: str, cycle: int, direction: int, elapsed: float
    ) -> None:
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        self.latest_cmd = {
            "phase": phase,
            "cycle": cycle,
            "direction": direction,
            "vx": msg.linear.x,
            "vy": msg.linear.y,
            "wz": msg.angular.z,
        }

        if rclpy.ok():
            self.cmd_pub.publish(msg)

        if not self.cmd_file.closed:
            self.cmd_writer.writerow(
                [
                    f"{elapsed:.9f}",
                    f"{stamp_to_sec(self.get_clock().now().to_msg()):.9f}",
                    phase,
                    cycle,
                    direction,
                    msg.linear.x,
                    msg.linear.y,
                    msg.angular.z,
                ]
            )

    def _finish(self, status: str) -> None:
        if self.finished:
            return
        self.finished = True
        elapsed = 0.0
        if self.motion_start_wall is not None:
            elapsed = time.monotonic() - self.motion_start_wall

        for _ in range(3):
            self._publish_command(0.0, status, 0, 0, elapsed)
            time.sleep(0.02)

        self.finished_at = datetime.now().isoformat(timespec="seconds")
        self._write_metadata(status)
        self._close_files()
        self.get_logger().info(f"Motion recorder finished with status '{status}': {self.output_dir}")

    def _close_files(self) -> None:
        for file_obj in (self.joint_file, self.cmd_file, self.odom_file):
            if not file_obj.closed:
                file_obj.close()

    def destroy_node(self) -> bool:
        self._close_files()
        return super().destroy_node()


def positive_float(parser: argparse.ArgumentParser, name: str, value: float) -> None:
    if value <= 0.0:
        parser.error(f"{name} must be > 0")


def nonnegative_float(parser: argparse.ArgumentParser, name: str, value: float) -> None:
    if value < 0.0:
        parser.error(f"{name} must be >= 0")


def parse_args(argv: Optional[list[str]] = None) -> Tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Drive RB-KAIROS forward/backward and record joint, command, and odom data."
    )
    parser.add_argument("--distance", type=float, default=1.0, help="one-way distance in meters")
    parser.add_argument("--speed", type=float, default=0.2, help="maximum forward speed in m/s")
    parser.add_argument("--accel", type=float, default=0.2, help="linear acceleration limit in m/s^2")
    parser.add_argument("--cycles", type=int, default=3, help="number of forward/backward cycles")
    parser.add_argument("--pause", type=float, default=0.5, help="zero-velocity pause after each leg")
    parser.add_argument("--rate", type=float, default=50.0, help="command publish rate in Hz")
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=15.0,
        help="seconds to wait for command subscriber and first data messages",
    )
    parser.add_argument("--data-dir", default="data", help="base directory for recorded output")
    parser.add_argument(
        "--cmd-topic",
        default="/mecanum_base_controller/reference_unstamped",
        help="Twist command topic for mecanum_base_controller",
    )
    parser.add_argument(
        "--dof-topic",
        default=DOF_TOPIC_DEFAULT,
        help="MujocoDofState topic containing qpos/qvel/qacc/qfrc data",
    )
    parser.add_argument(
        "--odom-topic",
        default="/base_ground_truth_odom",
        help="ground-truth odometry topic",
    )
    args, ros_args = parser.parse_known_args(argv)
    if ros_args:
        if "--ros-args" not in ros_args:
            parser.error("unknown arguments: " + " ".join(ros_args))
        ros_arg_index = ros_args.index("--ros-args")
        if ros_arg_index > 0:
            parser.error("unknown arguments: " + " ".join(ros_args[:ros_arg_index]))
        ros_args = ros_args[ros_arg_index:]

    positive_float(parser, "--distance", args.distance)
    positive_float(parser, "--speed", args.speed)
    positive_float(parser, "--accel", args.accel)
    positive_float(parser, "--rate", args.rate)
    nonnegative_float(parser, "--pause", args.pause)
    nonnegative_float(parser, "--startup-timeout", args.startup_timeout)
    if args.cycles <= 0:
        parser.error("--cycles must be > 0")

    return args, ros_args


def main(argv: Optional[list[str]] = None) -> None:
    args, ros_args = parse_args(argv)
    rclpy.init(args=ros_args)
    node = BackAndForthMotionRecorder(args)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node._finish("interrupted")
    finally:
        if not node.finished:
            node._finish("stopped")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
