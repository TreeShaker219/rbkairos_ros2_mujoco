#!/usr/bin/env python3
"""
Closed-loop x-position tracking validation for the RB-KAIROS MuJoCo setup.

Expected setup:
  ros2 launch rbkairos mujoco_mecanum_drive.launch.py

The arm posture is prepared manually before running this script. Use arm_preset_commander.py to set the arm posture quickly. Useful cases:
  arm_folded  - arm folded close to the base; compact COM, smaller inertia effect.
  arm_forward - arm stretched along +x; forward COM shift can change braking/overshoot.
  arm_left    - arm stretched toward +y; asymmetric load may create lateral/yaw coupling.
  arm_right   - arm stretched toward -y; opposite asymmetric load distribution.
  arm_up      - arm raised upward; higher COM may change contact/load distribution.

Example:
  ros2 run rbkairos position_tracking_validation.py \
    --case-name arm_folded \
    --target-distance 2.0 \
    --kp-x 0.8 \
    --rate 50 \
    --output-dir data/position_tracking_validation
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import re
import site
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import rclpy
from geometry_msgs.msg import Twist
from mujoco_ros2_control_msgs.msg import MujocoContactState, MujocoDofState
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


CMD_TOPIC_DEFAULT = "/mecanum_base_controller/reference_unstamped"
ODOM_TOPIC_DEFAULT = "/base_ground_truth_odom"
DOF_TOPIC_DEFAULT = "/mujoco_dof_state"
CONTACT_TOPIC_DEFAULT = "/mujoco_contact_state"


@dataclass
class OdomSample:
    t_ros: float
    t_experiment: float
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    wz: float
    x_error: float
    target_x: float


@dataclass
class CmdSample:
    t_ros: float
    t_experiment: float
    vx_cmd_raw: float
    vx_cmd_limited: float
    vy_cmd: float
    wz_cmd: float


@dataclass
class AccelSample:
    t_experiment: float
    accel_x: float


def stamp_to_sec(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def array_value(values: Any, index: int, default: Any = math.nan) -> Any:
    return values[index] if 0 <= index < len(values) else default


def array_slice(values: Any, start: int, count: int, default: float = math.nan) -> list[float]:
    result = list(values[start : start + count])
    if len(result) < count:
        result.extend([default] * (count - len(result)))
    return result


def sanitize_case_name(case_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_name.strip())
    return sanitized.strip("._-") or "unnamed_case"


def positive_float(parser: argparse.ArgumentParser, name: str, value: float) -> None:
    if value <= 0.0:
        parser.error(f"{name} must be > 0")


def nonnegative_float(parser: argparse.ArgumentParser, name: str, value: float) -> None:
    if value < 0.0:
        parser.error(f"{name} must be >= 0")


class PositionTrackingValidation(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("position_tracking_validation")

        self.args = args
        self.cmd_topic = args.cmd_topic
        self.odom_topic = args.odom_topic
        self.dof_topic = args.dof_topic
        self.contact_topic = args.contact_topic

        created_at = datetime.now().astimezone()
        timestamp = created_at.strftime("%Y%m%d_%H%M%S")
        self.case_name = sanitize_case_name(args.case_name)
        self.output_dir = Path(args.output_dir) / f"{timestamp}_{self.case_name}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.odom_path = self.output_dir / "odom.csv"
        self.cmd_path = self.output_dir / "cmd_vel.csv"
        self.joint_path = self.output_dir / "mujoco_dof_state.csv"
        self.joint_raw_path = self.output_dir / "mujoco_dof_state_raw.csv"
        self.contact_path = self.output_dir / "contact_forces.csv"
        self.metadata_path = self.output_dir / "metadata.json"
        self.metrics_path = self.output_dir / "metrics.json"

        self.odom_file = self.odom_path.open("w", newline="")
        self.cmd_file = self.cmd_path.open("w", newline="")
        self.joint_file = self.joint_path.open("w", newline="")
        self.joint_raw_file = self.joint_raw_path.open("w", newline="")
        self.contact_file = self.contact_path.open("w", newline="")
        self.odom_writer = csv.writer(self.odom_file)
        self.cmd_writer = csv.writer(self.cmd_file)
        self.joint_writer = csv.writer(self.joint_file)
        self.joint_raw_writer = csv.writer(self.joint_raw_file)
        self.contact_writer = csv.writer(self.contact_file)
        self._write_headers()

        self.created_at_wall = created_at.isoformat(timespec="seconds")
        self.start_wall_time: Optional[str] = None
        self.end_wall_time: Optional[str] = None
        self.start_wall_monotonic: Optional[float] = None
        self.end_wall_monotonic: Optional[float] = None
        self.start_ros_time: Optional[float] = None
        self.end_ros_time: Optional[float] = None
        self.experiment_time_source = "not_started"

        self.latest_odom_msg: Optional[Odometry] = None
        self.latest_odom: Optional[dict[str, float]] = None
        self.received_odom = False
        self.received_dof = False
        self.received_contact = False
        self.started = False
        self.finished = False
        self.stop_reason: Optional[str] = None
        self.settling_start_time: Optional[float] = None
        self.settling_time: Optional[float] = None
        self.last_cmd_limited = 0.0
        self.last_cmd_time: Optional[float] = None
        self.last_odom_for_accel: Optional[OdomSample] = None
        self.next_wait_log_wall = time.monotonic()

        self.x0: Optional[float] = None
        self.y0: Optional[float] = None
        self.yaw0: Optional[float] = None
        self.x_target: Optional[float] = None

        self.odom_samples: list[OdomSample] = []
        self.cmd_samples: list[CmdSample] = []
        self.accel_samples: list[AccelSample] = []

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self._on_odom, qos_profile_sensor_data
        )
        self.dof_sub = self.create_subscription(
            MujocoDofState, self.dof_topic, self._on_mujoco_dof_state, 50
        )
        self.contact_sub = self.create_subscription(
            MujocoContactState, self.contact_topic, self._on_mujoco_contact_state, 50
        )
        self.timer = self.create_timer(1.0 / float(args.rate), self._on_timer)

        self._write_metadata()
        self.get_logger().info(
            f"Waiting for first odometry sample on {self.odom_topic}. Output: {self.output_dir}"
        )

    def _write_headers(self) -> None:
        self.odom_writer.writerow(
            [
                "t_ros",
                "t_experiment",
                "x",
                "y",
                "yaw",
                "vx",
                "vy",
                "wz",
                "x_error",
                "target_x",
            ]
        )
        self.cmd_writer.writerow(
            [
                "t_ros",
                "t_experiment",
                "vx_cmd_raw",
                "vx_cmd_limited",
                "vy_cmd",
                "wz_cmd",
            ]
        )
        self.joint_writer.writerow(
            [
                "t_ros",
                "t_experiment",
                "header_frame_id",
                "dof_index",
                "dof_name",
                "qpos",
                "qvel",
                "qacc",
                "qfrc_constraint",
                "qfrc_applied",
                "qfrc_actuator",
                "qfrc_passive",
                "qfrc_bias",
                "qfrc_constraint_equality",
                "qfrc_constraint_friction",
                "qfrc_constraint_limit",
                "qfrc_constraint_contact",
                "qfrc_constraint_other",
                "nefc",
                "ncon",
            ]
        )
        self.joint_raw_writer.writerow(
            [
                "t_ros",
                "t_experiment",
                "header_frame_id",
                "dof_names",
                "nefc",
                "ncon",
                "qpos",
                "qvel",
                "qacc",
                "qfrc_applied",
                "qfrc_actuator",
                "qfrc_passive",
                "qfrc_bias",
                "qfrc_constraint",
                "qfrc_constraint_equality",
                "qfrc_constraint_friction",
                "qfrc_constraint_limit",
                "qfrc_constraint_contact",
                "qfrc_constraint_other",
            ]
        )
        self.contact_writer.writerow(
            [
                "t_ros",
                "t_experiment",
                "header_frame_id",
                "ncon",
                "contact_id",
                "geom1_id",
                "geom2_id",
                "geom1_name",
                "geom2_name",
                "dim",
                "efc_address",
                "dist",
                "position_x",
                "position_y",
                "position_z",
                "frame_3x3",
                "friction",
                "force_local_0",
                "force_local_1",
                "force_local_2",
                "torque_local_0",
                "torque_local_1",
                "torque_local_2",
                "force_local_6d",
            ]
        )

    def _on_odom(self, msg: Odometry) -> None:
        self.received_odom = True
        self.latest_odom_msg = msg

        pose = msg.pose.pose
        twist = msg.twist.twist
        yaw = yaw_from_quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        t_ros = stamp_to_sec(msg.header.stamp)
        self.latest_odom = {
            "t_ros": t_ros,
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "yaw": float(yaw),
            "vx": float(twist.linear.x),
            "vy": float(twist.linear.y),
            "wz": float(twist.angular.z),
        }

        if self.started and not self.finished:
            self._record_odom_sample()

    def _on_mujoco_dof_state(self, msg: MujocoDofState) -> None:
        self.received_dof = True
        if not self.started or self.finished:
            return

        t_ros = stamp_to_sec(msg.header.stamp)
        t_experiment = self._experiment_time_from_ros(t_ros)

        self.joint_raw_writer.writerow(
            [
                f"{t_ros:.9f}",
                f"{t_experiment:.9f}",
                msg.header.frame_id,
                json_cell(list(msg.dof_names)),
                int(msg.nefc),
                int(msg.ncon),
                json_cell(list(msg.qpos)),
                json_cell(list(msg.qvel)),
                json_cell(list(msg.qacc)),
                json_cell(list(msg.qfrc_applied)),
                json_cell(list(msg.qfrc_actuator)),
                json_cell(list(msg.qfrc_passive)),
                json_cell(list(msg.qfrc_bias)),
                json_cell(list(msg.qfrc_constraint)),
                json_cell(list(msg.qfrc_constraint_equality)),
                json_cell(list(msg.qfrc_constraint_friction)),
                json_cell(list(msg.qfrc_constraint_limit)),
                json_cell(list(msg.qfrc_constraint_contact)),
                json_cell(list(msg.qfrc_constraint_other)),
            ]
        )

        names = list(msg.dof_names)
        qpos = list(msg.qpos)
        qvel = list(msg.qvel)
        qacc = list(msg.qacc)
        applied = list(msg.qfrc_applied)
        actuator = list(msg.qfrc_actuator)
        passive = list(msg.qfrc_passive)
        bias = list(msg.qfrc_bias)
        constraint = list(msg.qfrc_constraint)
        constraint_equality = list(msg.qfrc_constraint_equality)
        constraint_friction = list(msg.qfrc_constraint_friction)
        constraint_limit = list(msg.qfrc_constraint_limit)
        constraint_contact = list(msg.qfrc_constraint_contact)
        constraint_other = list(msg.qfrc_constraint_other)
        dof_count = max(
            len(names),
            len(qpos),
            len(qvel),
            len(qacc),
            len(applied),
            len(actuator),
            len(passive),
            len(bias),
            len(constraint),
            len(constraint_equality),
            len(constraint_friction),
            len(constraint_limit),
            len(constraint_contact),
            len(constraint_other),
        )

        for i in range(dof_count):
            self.joint_writer.writerow(
                [
                    f"{t_ros:.9f}",
                    f"{t_experiment:.9f}",
                    msg.header.frame_id,
                    i,
                    names[i] if i < len(names) else f"dof_{i}",
                    array_value(qpos, i),
                    array_value(qvel, i),
                    array_value(qacc, i),
                    array_value(constraint, i),
                    array_value(applied, i),
                    array_value(actuator, i),
                    array_value(passive, i),
                    array_value(bias, i),
                    array_value(constraint_equality, i),
                    array_value(constraint_friction, i),
                    array_value(constraint_limit, i),
                    array_value(constraint_contact, i),
                    array_value(constraint_other, i),
                    int(msg.nefc),
                    int(msg.ncon),
                ]
            )

    def _on_mujoco_contact_state(self, msg: MujocoContactState) -> None:
        self.received_contact = True
        if not self.started or self.finished:
            return

        t_ros = stamp_to_sec(msg.header.stamp)
        t_experiment = self._experiment_time_from_ros(t_ros)
        ncon = int(msg.ncon)

        for i in range(ncon):
            position = array_slice(msg.position_xyz, 3 * i, 3)
            frame = array_slice(msg.frame_3x3, 9 * i, 9)
            friction = array_slice(msg.friction, 5 * i, 5)
            force = array_slice(msg.force_local_6d, 6 * i, 6)
            self.contact_writer.writerow(
                [
                    f"{t_ros:.9f}",
                    f"{t_experiment:.9f}",
                    msg.header.frame_id,
                    ncon,
                    array_value(msg.contact_id, i, i),
                    array_value(msg.geom1_id, i, -1),
                    array_value(msg.geom2_id, i, -1),
                    array_value(msg.geom1_name, i, ""),
                    array_value(msg.geom2_name, i, ""),
                    array_value(msg.dim, i, 0),
                    array_value(msg.efc_address, i, -1),
                    array_value(msg.dist, i),
                    position[0],
                    position[1],
                    position[2],
                    json_cell(frame),
                    json_cell(friction),
                    force[0],
                    force[1],
                    force[2],
                    force[3],
                    force[4],
                    force[5],
                    json_cell(force),
                ]
            )

    def _on_timer(self) -> None:
        if self.finished:
            return

        if not self.started:
            if self.received_odom:
                self._start_experiment()
            else:
                self._log_waiting_if_needed()
                return

        if self.latest_odom is None or self.x_target is None:
            return

        t_experiment = self._experiment_time_from_ros(self.latest_odom["t_ros"])
        if t_experiment >= float(self.args.max_duration):
            self._finish("max_duration")
            return

        x_current = self.latest_odom["x"]
        x_error = self.x_target - x_current
        vx_raw = float(self.args.kp_x) * x_error
        vx_clamped = self._apply_vx_limit(vx_raw)
        vx_limited = self._apply_accel_limit(vx_clamped, t_experiment)

        self._publish_command(vx_raw, vx_limited, 0.0, 0.0, t_experiment)

        settled = (
            abs(x_error) < float(self.args.position_tolerance)
            and abs(self.latest_odom["vx"]) < float(self.args.velocity_tolerance)
        )
        if settled:
            if self.settling_start_time is None:
                self.settling_start_time = t_experiment
            if t_experiment - self.settling_start_time >= float(self.args.settling_duration):
                self.settling_time = self.settling_start_time
                self._finish("settled")
        else:
            self.settling_start_time = None

    def _apply_vx_limit(self, vx_target: float) -> float:
        max_vx = float(self.args.max_vx)
        if max_vx <= 0.0:
            return vx_target
        return clamp(vx_target, -max_vx, max_vx)

    def _log_waiting_if_needed(self) -> None:
        now = time.monotonic()
        if now >= self.next_wait_log_wall:
            self.get_logger().info(f"Waiting for odometry on {self.odom_topic}...")
            self.next_wait_log_wall = now + 2.0

    def _start_experiment(self) -> None:
        if self.latest_odom is None:
            return

        self.started = True
        self.start_wall_monotonic = time.monotonic()
        self.start_wall_time = datetime.now().astimezone().isoformat(timespec="seconds")
        self.start_ros_time = self.latest_odom["t_ros"]
        self.experiment_time_source = "relative_to_base_ground_truth_odom_header_stamp"

        self.x0 = self.latest_odom["x"]
        self.y0 = self.latest_odom["y"]
        self.yaw0 = self.latest_odom["yaw"]
        self.x_target = self.x0 + float(self.args.target_distance)
        self.last_cmd_time = 0.0
        self.last_cmd_limited = 0.0
        self._write_metadata()
        self._record_odom_sample()

        if self.cmd_pub.get_subscription_count() == 0:
            self.get_logger().warn(
                f"No subscribers currently visible on {self.cmd_topic}; commands will still be published."
            )
        if not self.received_dof:
            self.get_logger().warn(
                f"No {self.dof_topic} samples received yet; mujoco_dof_state.csv will start when data arrives."
            )

        self.get_logger().info(
            f"Starting case '{self.case_name}': x0={self.x0:.3f}, "
            f"target_x={self.x_target:.3f}, distance={self.args.target_distance:.3f} m"
        )

    def _record_odom_sample(self) -> None:
        if self.latest_odom is None or self.x_target is None:
            return

        t_ros = self.latest_odom["t_ros"]
        t_experiment = self._experiment_time_from_ros(t_ros)
        x_error = self.x_target - self.latest_odom["x"]
        sample = OdomSample(
            t_ros=t_ros,
            t_experiment=t_experiment,
            x=self.latest_odom["x"],
            y=self.latest_odom["y"],
            yaw=self.latest_odom["yaw"],
            vx=self.latest_odom["vx"],
            vy=self.latest_odom["vy"],
            wz=self.latest_odom["wz"],
            x_error=x_error,
            target_x=self.x_target,
        )
        self.odom_samples.append(sample)
        self.odom_writer.writerow(
            [
                f"{sample.t_ros:.9f}",
                f"{sample.t_experiment:.9f}",
                sample.x,
                sample.y,
                sample.yaw,
                sample.vx,
                sample.vy,
                sample.wz,
                sample.x_error,
                sample.target_x,
            ]
        )

        if self.last_odom_for_accel is not None:
            dt = sample.t_experiment - self.last_odom_for_accel.t_experiment
            if dt > 1e-9:
                accel = (sample.vx - self.last_odom_for_accel.vx) / dt
                self.accel_samples.append(AccelSample(sample.t_experiment, accel))
        self.last_odom_for_accel = sample

    def _apply_accel_limit(self, vx_target: float, t_experiment: float) -> float:
        max_accel = float(self.args.max_accel)
        if max_accel <= 0.0 or self.last_cmd_time is None:
            self.last_cmd_time = t_experiment
            self.last_cmd_limited = vx_target
            return vx_target

        dt = max(0.0, t_experiment - self.last_cmd_time)
        max_delta = max_accel * dt
        vx_limited = clamp(
            vx_target,
            self.last_cmd_limited - max_delta,
            self.last_cmd_limited + max_delta,
        )
        self.last_cmd_time = t_experiment
        self.last_cmd_limited = vx_limited
        return vx_limited

    def _publish_command(
        self,
        vx_cmd_raw: float,
        vx_cmd_limited: float,
        vy_cmd: float,
        wz_cmd: float,
        t_experiment: float,
        record: bool = True,
    ) -> None:
        msg = Twist()
        msg.linear.x = float(vx_cmd_limited)
        msg.linear.y = float(vy_cmd)
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(wz_cmd)
        self.cmd_pub.publish(msg)

        t_ros = self.latest_odom["t_ros"] if self.latest_odom is not None else self._node_now_sec()
        if record and self.started and not self.cmd_file.closed:
            sample = CmdSample(t_ros, t_experiment, vx_cmd_raw, vx_cmd_limited, vy_cmd, wz_cmd)
            self.cmd_samples.append(sample)
            self.cmd_writer.writerow(
                [
                    f"{sample.t_ros:.9f}",
                    f"{sample.t_experiment:.9f}",
                    sample.vx_cmd_raw,
                    sample.vx_cmd_limited,
                    sample.vy_cmd,
                    sample.wz_cmd,
                ]
            )

    def _publish_zero_commands(self, count: int = 8, interval_sec: float = 0.025) -> None:
        t_experiment = self._current_experiment_time()
        for _ in range(count):
            if rclpy.ok():
                self._publish_command(0.0, 0.0, 0.0, 0.0, t_experiment, record=self.started)
            time.sleep(interval_sec)

    def _finish(self, reason: str) -> None:
        if self.finished:
            return

        self.stop_reason = reason
        self._publish_zero_commands()
        self.finished = True
        self.end_wall_monotonic = time.monotonic()
        self.end_wall_time = datetime.now().astimezone().isoformat(timespec="seconds")
        self.end_ros_time = self._node_now_sec()
        self._close_csv_files()

        metrics = self._compute_metrics()
        self._write_json(self.metrics_path, metrics)
        plot_status = self._write_plots()
        self._write_metadata(plot_status=plot_status)
        self.get_logger().info(f"Finished with stop_reason='{reason}'. Output: {self.output_dir}")

    def _experiment_time_from_ros(self, t_ros: float) -> float:
        if self.start_ros_time is not None:
            return max(0.0, t_ros - self.start_ros_time)
        if self.start_wall_monotonic is not None:
            return max(0.0, time.monotonic() - self.start_wall_monotonic)
        return 0.0

    def _current_experiment_time(self) -> float:
        if self.latest_odom is not None:
            return self._experiment_time_from_ros(self.latest_odom["t_ros"])
        if self.start_wall_monotonic is not None:
            return max(0.0, time.monotonic() - self.start_wall_monotonic)
        return 0.0

    def _node_now_sec(self) -> float:
        return stamp_to_sec(self.get_clock().now().to_msg())

    def _close_csv_files(self) -> None:
        for file_obj in (
            self.odom_file,
            self.cmd_file,
            self.joint_file,
            self.joint_raw_file,
            self.contact_file,
        ):
            if not file_obj.closed:
                file_obj.close()

    def _compute_metrics(self) -> dict[str, Any]:
        if not self.odom_samples:
            return {
                "stop_reason": self.stop_reason,
                "duration": 0.0,
                "error": "no odometry samples were recorded",
            }

        duration = self.odom_samples[-1].t_experiment
        final_error = self.odom_samples[-1].x_error
        errors = [sample.x_error for sample in self.odom_samples]
        xs = [sample.x for sample in self.odom_samples]
        vxs = [sample.vx for sample in self.odom_samples]
        cmd_vxs = [sample.vx_cmd_limited for sample in self.cmd_samples]
        accels = [sample.accel_x for sample in self.accel_samples]

        target_distance = float(self.args.target_distance)
        direction = 1.0 if target_distance >= 0.0 else -1.0
        target_x = self.x_target if self.x_target is not None else xs[-1]
        max_overshoot_x = max(0.0, max(direction * (x - target_x) for x in xs))

        rise_time_10_90 = self._compute_rise_time_10_90(direction)
        rms_tracking_error = math.sqrt(sum(error * error for error in errors) / len(errors))

        return {
            "final_position_error": final_error,
            "max_abs_position_error": max(abs(error) for error in errors),
            "max_overshoot_x": max_overshoot_x,
            "rise_time_10_90": rise_time_10_90,
            "settling_time": self.settling_time,
            "max_abs_vx": max(abs(vx) for vx in vxs) if vxs else 0.0,
            "max_abs_vx_cmd": max(abs(vx) for vx in cmd_vxs) if cmd_vxs else 0.0,
            "max_abs_accel_estimated_from_odom": max(abs(accel) for accel in accels)
            if accels
            else 0.0,
            "rms_tracking_error": rms_tracking_error,
            "duration": duration,
            "stop_reason": self.stop_reason,
        }

    def _compute_rise_time_10_90(self, direction: float) -> Optional[float]:
        if self.x0 is None:
            return None
        distance = abs(float(self.args.target_distance))
        if distance <= 1e-9:
            return None

        threshold_10 = 0.1 * distance
        threshold_90 = 0.9 * distance
        t10 = None
        t90 = None
        for sample in self.odom_samples:
            progress = direction * (sample.x - self.x0)
            if t10 is None and progress >= threshold_10:
                t10 = sample.t_experiment
            if t90 is None and progress >= threshold_90:
                t90 = sample.t_experiment
                break
        if t10 is None or t90 is None:
            return None
        return max(0.0, t90 - t10)

    def _write_plots(self) -> dict[str, Any]:
        if not self.odom_samples:
            return {"generated": False, "reason": "no odometry samples"}

        plt, import_status = self._import_pyplot()
        if plt is None:
            return {
                "generated": False,
                "reason": "matplotlib import failed",
                "matplotlib_import": import_status,
            }

        t = [sample.t_experiment for sample in self.odom_samples]
        x = [sample.x for sample in self.odom_samples]
        y = [sample.y for sample in self.odom_samples]
        target = [sample.target_x for sample in self.odom_samples]
        x_error = [sample.x_error for sample in self.odom_samples]
        vx = [sample.vx for sample in self.odom_samples]
        cmd_t = [sample.t_experiment for sample in self.cmd_samples]
        cmd_vx = [sample.vx_cmd_limited for sample in self.cmd_samples]
        accel_t = [sample.t_experiment for sample in self.accel_samples]
        accel = [sample.accel_x for sample in self.accel_samples]

        self._plot_xy(
            plt,
            self.output_dir / "x_position_vs_time.png",
            t,
            x,
            "time [s]",
            "x [m]",
            "x position vs time",
            second_y=target,
            second_label="target_x",
            first_label="x",
        )
        self._plot_xy(
            plt,
            self.output_dir / "x_error_vs_time.png",
            t,
            x_error,
            "time [s]",
            "x error [m]",
            "x error vs time",
            first_label="x_target - x",
        )
        self._plot_xy(
            plt,
            self.output_dir / "vx_vs_time.png",
            t,
            vx,
            "time [s]",
            "vx [m/s]",
            "measured and commanded vx",
            second_x=cmd_t,
            second_y=cmd_vx,
            first_label="measured vx",
            second_label="commanded vx",
        )
        self._plot_xy(
            plt,
            self.output_dir / "estimated_accel_vs_time.png",
            accel_t,
            accel,
            "time [s]",
            "estimated accel [m/s^2]",
            "estimated acceleration from odom vx",
            first_label="estimated accel",
        )

        fig, ax = plt.subplots()
        ax.plot(x, y, label="trajectory")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("xy trajectory")
        ax.axis("equal")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "xy_trajectory.png", dpi=150)
        plt.close(fig)

        return {
            "generated": True,
            "matplotlib_import": import_status,
            "files": [
                "x_position_vs_time.png",
                "x_error_vs_time.png",
                "vx_vs_time.png",
                "estimated_accel_vs_time.png",
                "xy_trajectory.png",
            ],
        }

    def _import_pyplot(self) -> tuple[Optional[Any], dict[str, Any]]:
        plt, reason = self._try_import_pyplot()
        if plt is not None:
            return plt, {"mode": "default"}

        original_path = list(sys.path)
        removed_modules = self._remove_matplotlib_numpy_modules()
        self._remove_user_site_from_path()
        plt, fallback_reason = self._try_import_pyplot()
        if plt is not None:
            return plt, {
                "mode": "without_user_site",
                "initial_error": reason,
            }

        sys.path = original_path
        sys.modules.update(removed_modules)
        return None, {
            "mode": "failed",
            "initial_error": reason,
            "fallback_error": fallback_reason,
        }

    def _try_import_pyplot(self) -> tuple[Optional[Any], str]:
        stderr_capture = io.StringIO()
        try:
            with contextlib.redirect_stderr(stderr_capture):
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            return plt, ""
        except Exception as exc:
            stderr_text = stderr_capture.getvalue().strip()
            reason = str(exc) or stderr_text or exc.__class__.__name__
            return None, reason

    def _remove_user_site_from_path(self) -> None:
        user_sites: set[str] = set()
        for value in (site.USER_SITE, site.getusersitepackages()):
            if not value:
                continue
            if isinstance(value, str):
                user_sites.add(str(Path(value).resolve()))
            else:
                user_sites.update(str(Path(path).resolve()) for path in value)

        sys.path = [
            path
            for path in sys.path
            if not path or str(Path(path).resolve()) not in user_sites
        ]

    def _remove_matplotlib_numpy_modules(self) -> dict[str, Any]:
        removed = {}
        for name in list(sys.modules):
            if name == "matplotlib" or name.startswith("matplotlib."):
                removed[name] = sys.modules.pop(name)
            elif name == "numpy" or name.startswith("numpy."):
                removed[name] = sys.modules.pop(name)
        return removed

    def _plot_xy(
        self,
        plt: Any,
        path: Path,
        first_x: list[float],
        first_y: list[float],
        xlabel: str,
        ylabel: str,
        title: str,
        first_label: str,
        second_x: Optional[list[float]] = None,
        second_y: Optional[list[float]] = None,
        second_label: Optional[str] = None,
    ) -> None:
        fig, ax = plt.subplots()
        ax.plot(first_x, first_y, label=first_label)
        if second_y is not None:
            ax.plot(second_x if second_x is not None else first_x, second_y, label=second_label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _write_metadata(self, plot_status: Optional[dict[str, Any]] = None) -> None:
        metadata = {
            "case_name": self.case_name,
            "target_distance": self.args.target_distance,
            "x0": self.x0,
            "y0": self.y0,
            "yaw0": self.yaw0,
            "x_target": self.x_target,
            "controller_parameters": {
                "kp_x": self.args.kp_x,
                "max_vx": self.args.max_vx,
                "max_accel": self.args.max_accel,
                "rate": self.args.rate,
                "position_tolerance": self.args.position_tolerance,
                "velocity_tolerance": self.args.velocity_tolerance,
                "settling_duration": self.args.settling_duration,
                "max_duration": self.args.max_duration,
            },
            "topics": {
                "cmd_vel": self.cmd_topic,
                "ground_truth_odom": self.odom_topic,
                "mujoco_dof_state": self.dof_topic,
                "mujoco_dof_state_type": "mujoco_ros2_control_msgs/msg/MujocoDofState",
                "mujoco_contact_state": self.contact_topic,
                "mujoco_contact_state_type": "mujoco_ros2_control_msgs/msg/MujocoContactState",
            },
            "wall_time": {
                "created": self.created_at_wall,
                "start": self.start_wall_time,
                "end": self.end_wall_time,
            },
            "ros_time_usage": {
                "experiment_time_source": self.experiment_time_source,
                "start_ros_time": self.start_ros_time,
                "end_ros_time": self.end_ros_time,
                "cmd_vel_t_ros_source": "latest odom header stamp used for the control update",
                "odom_t_ros_source": "nav_msgs/Odometry.header.stamp",
                "mujoco_dof_state_t_ros_source": "MujocoDofState.header.stamp",
                "mujoco_contact_state_t_ros_source": "MujocoContactState.header.stamp",
            },
            "output_files": {
                "odom": str(self.odom_path),
                "cmd_vel": str(self.cmd_path),
                "mujoco_dof_state": str(self.joint_path),
                "mujoco_dof_state_raw": str(self.joint_raw_path),
                "contact_forces": str(self.contact_path),
                "metadata": str(self.metadata_path),
                "metrics": str(self.metrics_path),
            },
            "stop_reason": self.stop_reason,
            "plot_status": plot_status,
            "notes": [
                "The robot arm posture is manually prepared before this script starts.",
                "The script only commands mobile-base x velocity and does not move the arm.",
                "Compare output curves across case names to inspect COM/inertia effects.",
                "mujoco_dof_state.csv is a tidy table with one MuJoCo nv DOF per row; mujoco_dof_state_raw.csv preserves each full MujocoDofState message with JSON array cells.",
                "qfrc_constraint_* columns are mj_mulJacTVec projections of efc_force grouped by MuJoCo constraint type.",
                "contact_forces.csv contains one row per MuJoCo contact and force_local_6d from mj_contactForce().",
                "The base ground-truth odometry pose is in the world frame; its twist is published from MuJoCo local body velocity.",
            ],
        }
        self._write_json(self.metadata_path, metadata)

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, indent=2, sort_keys=True)
            file_obj.write("\n")

    def destroy_node(self) -> bool:
        if not self.finished:
            self._close_csv_files()
        return super().destroy_node()


def parse_args(argv: Optional[list[str]] = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Closed-loop x-position tracking validation for RB-KAIROS MuJoCo."
    )
    parser.add_argument("--target-distance", type=float, default=2.0, help="world x displacement [m]")
    parser.add_argument("--kp-x", type=float, default=0.8, help="proportional x gain")
    parser.add_argument(
        "--max-vx",
        type=float,
        default=0.0,
        help="absolute x command limit [m/s]; set 0 to disable",
    )
    parser.add_argument(
        "--max-accel",
        type=float,
        default=0.0,
        help="slew-rate limit for commanded vx [m/s^2]; set 0 to disable",
    )
    parser.add_argument("--rate", type=float, default=50.0, help="control loop rate [Hz]")
    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=0.02,
        help="settling position tolerance [m]",
    )
    parser.add_argument(
        "--velocity-tolerance",
        type=float,
        default=0.02,
        help="settling velocity tolerance [m/s]",
    )
    parser.add_argument(
        "--settling-duration",
        type=float,
        default=1.0,
        help="continuous settled duration required before stopping [s]",
    )
    parser.add_argument("--max-duration", type=float, default=30.0, help="maximum experiment time [s]")
    parser.add_argument("--case-name", default="arm_folded", help="manual arm posture case name")
    parser.add_argument(
        "--output-dir",
        default="data/position_tracking_validation",
        help="base output directory",
    )
    parser.add_argument("--cmd-topic", default=CMD_TOPIC_DEFAULT, help="Twist command topic")
    parser.add_argument("--odom-topic", default=ODOM_TOPIC_DEFAULT, help="ground-truth odom topic")
    parser.add_argument(
        "--dof-topic",
        default=DOF_TOPIC_DEFAULT,
        help="MujocoDofState topic containing MuJoCo qpos/qvel/qacc/qfrc data",
    )
    parser.add_argument(
        "--contact-topic",
        default=CONTACT_TOPIC_DEFAULT,
        help="MujocoContactState topic containing per-contact mj_contactForce data",
    )

    args, ros_args = parser.parse_known_args(argv)
    if ros_args:
        if "--ros-args" not in ros_args:
            parser.error("unknown arguments: " + " ".join(ros_args))
        ros_arg_index = ros_args.index("--ros-args")
        if ros_arg_index > 0:
            parser.error("unknown arguments: " + " ".join(ros_args[:ros_arg_index]))
        ros_args = ros_args[ros_arg_index:]

    if abs(args.target_distance) <= 1e-9:
        parser.error("--target-distance must be non-zero")
    positive_float(parser, "--kp-x", args.kp_x)
    nonnegative_float(parser, "--max-vx", args.max_vx)
    nonnegative_float(parser, "--max-accel", args.max_accel)
    positive_float(parser, "--rate", args.rate)
    positive_float(parser, "--position-tolerance", args.position_tolerance)
    positive_float(parser, "--velocity-tolerance", args.velocity_tolerance)
    nonnegative_float(parser, "--settling-duration", args.settling_duration)
    positive_float(parser, "--max-duration", args.max_duration)
    if not args.case_name.strip():
        parser.error("--case-name must not be empty")

    return args, ros_args


def main(argv: Optional[list[str]] = None) -> None:
    args, ros_args = parse_args(argv)
    rclpy.init(args=ros_args)
    node = PositionTrackingValidation(args)
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
