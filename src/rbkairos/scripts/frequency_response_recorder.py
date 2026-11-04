#!/usr/bin/env python3
"""
Frequency-response recorder for the RB-KAIROS MuJoCo mecanum base.

Expected setup:
  ros2 launch rbkairos mujoco_mecanum_drive.launch.py

Example:
  ros2 run rbkairos frequency_response_recorder.py \
    --environment sim \
    --axis x \
    --amplitude 0.03 \
    --frequencies 0.12 0.25 \
    --cycles 4 \
    --discard-cycles 1 \
    --publish-rate 100 \
    --output-dir data/frequency_response
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import rclpy
from geometry_msgs.msg import Twist
from mujoco_ros2_control_msgs.msg import MujocoContactState, MujocoDofState
from nav_msgs.msg import Odometry
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.parameter import parameter_value_to_python
from rclpy.qos import qos_profile_sensor_data


CMD_TOPIC_DEFAULT = "/mecanum_base_controller/reference_unstamped"
ODOM_TOPIC_DEFAULT = "/base_ground_truth_odom"
DOF_TOPIC_DEFAULT = "/mujoco_dof_state"
CONTACT_TOPIC_DEFAULT = "/mujoco_contact_state"

WHEEL_DOF_NAMES = [
    "left_front_wheel_joint",
    "right_front_wheel_joint",
    "left_back_wheel_joint",
    "right_back_wheel_joint",
]
PID_DIAGNOSTIC_SCHEMA_VERSION = 2
FRF_PID_NONBINDING_EVIDENCE_SCHEMA = "rbkairos_frf_pid_nonbinding_evidence_v1"
PID_NUMERIC_FIELDS = (
    ("velocity_pid_target", "velocity_target"),
    ("velocity_pid_actual", "velocity_actual"),
    ("velocity_pid_error", "velocity_error"),
    ("velocity_pid_p_term", "p_term"),
    ("velocity_pid_i_term", "i_term"),
    ("velocity_pid_candidate_i_term", "candidate_i_term"),
    (
        "velocity_pid_peak_abs_candidate_i_term",
        "peak_abs_candidate_i_term",
    ),
    ("velocity_pid_d_term", "d_term"),
    ("velocity_pid_raw_output", "pid_raw_output"),
    ("velocity_pid_effort_clamped_output", "effort_clamped_output"),
    ("velocity_pid_kp", "kp"),
    ("velocity_pid_ki", "ki"),
    ("velocity_pid_kd", "kd"),
    ("velocity_pid_i_max", "i_max"),
    ("velocity_pid_i_min", "i_min"),
    ("velocity_pid_effort_limit", "effort_limit_nm"),
)
PID_BOOL_FIELDS = (
    ("velocity_pid_i_clamp_active", "i_clamp_active"),
    ("velocity_pid_effort_clamp_active", "effort_clamp_active"),
    ("velocity_pid_i_clamp_ever", "i_clamp_ever"),
    ("velocity_pid_effort_clamp_ever", "effort_clamp_ever"),
    ("velocity_pid_antiwindup", "antiwindup"),
)
PID_INTEGER_FIELDS = (
    ("velocity_pid_control_update_count", "control_update_count"),
)
PID_CSV_FIELDS = (
    "t_experiment",
    "header_stamp_s",
    "joint_name",
    "velocity_target",
    "velocity_actual",
    "velocity_error",
    "p_term",
    "i_term",
    "candidate_i_term",
    "peak_abs_candidate_i_term",
    "d_term",
    "pid_raw_output",
    "effort_clamped_output",
    "qfrc_applied",
    "i_clamp_active",
    "effort_clamp_active",
    "i_clamp_ever",
    "effort_clamp_ever",
    "control_update_count",
    "kp",
    "ki",
    "kd",
    "i_max",
    "i_min",
    "antiwindup",
    "effort_limit_nm",
)
WHEEL_QFRC_SUMMARY_FIELDS = [
    "t_ros",
    "t_experiment",
    "header_frame_id",
    "wheel_count",
    "max_abs_qfrc_applied",
    "max_abs_qfrc_wheel",
] + [f"{name}_qfrc_applied" for name in WHEEL_DOF_NAMES] + [
    "nefc",
    "ncon",
]
CONTACT_FORCE_SUMMARY_FIELDS = [
    "t_ros",
    "t_experiment",
    "header_frame_id",
    "ncon",
    "contact_count",
    "sum_force_norm_sq",
    "max_force_norm",
    "normal_abs_sum",
    "tangent_norm_sum",
]

ROBOT_DESCRIPTION_NODE_DEFAULT = "/robot_state_publisher"
CONTROLLER_MANAGER_NODE_DEFAULT = "/controller_manager"
BASE_CONTROLLER_NODE_DEFAULT = "/mecanum_base_controller"
SIM_NODE_DEFAULT = "/mujoco_sim_ros2_node"


@dataclass
class RunContext:
    run_id: str
    frequency_hz: float
    omega_rad_s: float
    run_dir: Path
    planned_duration: float
    max_duration: float
    active: bool = False
    completed: bool = False
    valid: bool = True
    reject_reason: str = ""
    pre_settle_start_wall: Optional[float] = None
    start_wall_time: Optional[str] = None
    end_wall_time: Optional[str] = None
    start_wall_monotonic: Optional[float] = None
    end_wall_monotonic: Optional[float] = None
    start_ros_time: Optional[float] = None
    end_ros_time: Optional[float] = None
    x_start: Optional[float] = None
    y_start: Optional[float] = None
    yaw_start: Optional[float] = None
    x_final: Optional[float] = None
    y_final: Optional[float] = None
    yaw_final: Optional[float] = None
    odom_samples: int = 0
    cmd_samples: int = 0
    dof_samples: int = 0
    wheel_qfrc_summary_rows: int = 0
    wheel_velocity_pid_rows: int = 0
    velocity_pid_contract_errors: int = 0
    pid_nonbinding_evidence: Any = None
    contact_messages: int = 0
    contact_force_summary_rows: int = 0
    contact_rows: int = 0


def stamp_to_sec(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def array_value(values: Any, index: int, default: Any = math.nan) -> Any:
    return values[index] if 0 <= index < len(values) else default


def array_slice(values: Any, start: int, count: int, default: float = math.nan) -> list[float]:
    result = list(values[start:start + count])
    if len(result) < count:
        result.extend([default] * (count - len(result)))
    return result


class PidDiagnosticContractError(ValueError):
    """Raised when the versioned wheel PID diagnostic contract is invalid."""


def decode_velocity_pid_diagnostics(message: Any) -> dict[str, dict[str, Any]]:
    """Decode and cross-check the read-only controller diagnostics in one DOF sample."""
    version = int(getattr(message, "velocity_pid_diagnostic_schema_version", -1))
    if version != PID_DIAGNOSTIC_SCHEMA_VERSION:
        raise PidDiagnosticContractError(
            f"velocity PID diagnostic schema {version} != {PID_DIAGNOSTIC_SCHEMA_VERSION}"
        )

    names = list(getattr(message, "velocity_pid_joint_names", []))
    if len(names) != len(set(names)) or set(names) != set(WHEEL_DOF_NAMES):
        raise PidDiagnosticContractError(
            "velocity PID diagnostics must contain exactly the four wheel joints"
        )
    count = len(names)
    decoded: dict[str, dict[str, Any]] = {name: {} for name in names}

    for message_field, record_field in PID_NUMERIC_FIELDS:
        values = list(getattr(message, message_field, []))
        if len(values) != count:
            raise PidDiagnosticContractError(
                f"{message_field} has {len(values)} values, expected {count}"
            )
        for name, raw_value in zip(names, values):
            value = float(raw_value)
            if not math.isfinite(value):
                raise PidDiagnosticContractError(
                    f"non-finite {message_field} for {name}"
                )
            decoded[name][record_field] = value

    for message_field, record_field in PID_BOOL_FIELDS:
        values = list(getattr(message, message_field, []))
        if len(values) != count:
            raise PidDiagnosticContractError(
                f"{message_field} has {len(values)} values, expected {count}"
            )
        for name, raw_value in zip(names, values):
            decoded[name][record_field] = bool(raw_value)

    for message_field, record_field in PID_INTEGER_FIELDS:
        values = list(getattr(message, message_field, []))
        if len(values) != count:
            raise PidDiagnosticContractError(
                f"{message_field} has {len(values)} values, expected {count}"
            )
        for name, raw_value in zip(names, values):
            value = int(raw_value)
            if value <= 0:
                raise PidDiagnosticContractError(
                    f"invalid {message_field} for {name}"
                )
            decoded[name][record_field] = value

    dof_names = list(getattr(message, "dof_names", []))
    qfrc_applied = list(getattr(message, "qfrc_applied", []))
    if len(qfrc_applied) != len(dof_names):
        raise PidDiagnosticContractError("qfrc_applied length does not match dof_names")
    dof_index = {name: index for index, name in enumerate(dof_names)}

    for name, record in decoded.items():
        if name not in dof_index:
            raise PidDiagnosticContractError(f"PID joint {name} is absent from dof_names")
        qfrc = float(qfrc_applied[dof_index[name]])
        if not math.isfinite(qfrc):
            raise PidDiagnosticContractError(f"non-finite qfrc_applied for {name}")
        record["qfrc_applied"] = qfrc
        identities = (
            (
                record["velocity_error"],
                record["velocity_target"] - record["velocity_actual"],
                "velocity error",
            ),
            (record["p_term"], record["kp"] * record["velocity_error"], "P term"),
            (
                record["pid_raw_output"],
                record["p_term"] + record["i_term"] + record["d_term"],
                "P+I+D",
            ),
        )
        for actual, expected, label in identities:
            if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
                raise PidDiagnosticContractError(f"{label} identity failed for {name}")
        if record["i_min"] > record["i_max"] or not (
            record["i_min"] - 1e-9 <= record["i_term"] <= record["i_max"] + 1e-9
        ):
            raise PidDiagnosticContractError(f"I term exceeds its limits for {name}")
        peak = record["peak_abs_candidate_i_term"]
        if peak < 0.0 or abs(record["candidate_i_term"]) > peak + 1e-12:
            raise PidDiagnosticContractError(
                f"candidate I-term peak is inconsistent for {name}"
            )
        if record["i_clamp_active"] and not record["i_clamp_ever"]:
            raise PidDiagnosticContractError(f"I clamp latch is inconsistent for {name}")
        if record["effort_clamp_active"] and not record["effort_clamp_ever"]:
            raise PidDiagnosticContractError(
                f"effort clamp latch is inconsistent for {name}"
            )
        effort_limit = record["effort_limit_nm"]
        if effort_limit <= 0.0:
            raise PidDiagnosticContractError(f"invalid effort limit for {name}")
        expected_clamped = max(-effort_limit, min(record["pid_raw_output"], effort_limit))
        if not math.isclose(
            record["effort_clamped_output"], expected_clamped, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise PidDiagnosticContractError(f"effort clamp identity failed for {name}")
        expected_effort_clamp = not math.isclose(
            record["pid_raw_output"], expected_clamped, rel_tol=1e-12, abs_tol=1e-12
        )
        if record["effort_clamp_active"] != expected_effort_clamp:
            raise PidDiagnosticContractError(f"effort clamp flag mismatch for {name}")
        if not math.isclose(
            record["effort_clamped_output"], qfrc, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise PidDiagnosticContractError(
                f"effort_clamped_output != qfrc_applied for {name}"
            )

    return {name: decoded[name] for name in WHEEL_DOF_NAMES}


class PidNonbindingEvidence:
    """Aggregate controller-rate latches without relying on every ROS sample."""

    def __init__(self, frequency_hz: float) -> None:
        self.frequency_hz = float(frequency_hz)
        self.message_samples = 0
        self.per_wheel: dict[str, dict[str, Any]] = {}

    def update(self, diagnostics: dict[str, dict[str, Any]]) -> None:
        if set(diagnostics) != set(WHEEL_DOF_NAMES):
            raise PidDiagnosticContractError(
                "nonbinding evidence must contain exactly the four wheel joints"
            )
        for wheel in WHEEL_DOF_NAMES:
            record = diagnostics[wheel]
            current = {
                "control_update_count": int(record["control_update_count"]),
                "max_abs_candidate_i_term_nm": float(
                    record["peak_abs_candidate_i_term"]
                ),
                "i_clamp_ever": bool(record["i_clamp_ever"]),
                "effort_clamp_ever": bool(record["effort_clamp_ever"]),
            }
            previous = self.per_wheel.get(wheel)
            if previous is not None:
                if current["control_update_count"] < previous["control_update_count"]:
                    raise PidDiagnosticContractError(
                        f"controller update count decreased for {wheel}"
                    )
                if (
                    current["max_abs_candidate_i_term_nm"] + 1e-12
                    < previous["max_abs_candidate_i_term_nm"]
                ):
                    raise PidDiagnosticContractError(
                        f"candidate I-term peak decreased for {wheel}"
                    )
                if previous["i_clamp_ever"] and not current["i_clamp_ever"]:
                    raise PidDiagnosticContractError(
                        f"I clamp latch reverted for {wheel}"
                    )
                if previous["effort_clamp_ever"] and not current["effort_clamp_ever"]:
                    raise PidDiagnosticContractError(
                        f"effort clamp latch reverted for {wheel}"
                    )
            self.per_wheel[wheel] = current
        self.message_samples += 1

    def to_dict(self) -> dict[str, Any]:
        complete = self.message_samples > 0 and set(self.per_wheel) == set(WHEEL_DOF_NAMES)
        return {
            "schema": FRF_PID_NONBINDING_EVIDENCE_SCHEMA,
            "frequency_hz": self.frequency_hz,
            "complete": complete,
            "controller_rate_latched": True,
            "diagnostic_schema_version": PID_DIAGNOSTIC_SCHEMA_VERSION,
            "message_samples": self.message_samples,
            "per_wheel": {
                wheel: dict(self.per_wheel[wheel])
                for wheel in WHEEL_DOF_NAMES
                if wheel in self.per_wheel
            },
        }


def sanitize_token(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return sanitized.strip("._-") or "unnamed"


def format_float_token(value: float, precision: int = 3) -> str:
    text = f"{value:.{precision}f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def format_frequency_run_id(frequency_hz: float) -> str:
    return f"f_{frequency_hz:.3f}".replace(".", "p") + "Hz"


def positive_float(parser: argparse.ArgumentParser, name: str, value: float) -> None:
    if value <= 0.0:
        parser.error(f"{name} must be > 0")


def nonnegative_float(parser: argparse.ArgumentParser, name: str, value: float) -> None:
    if value < 0.0:
        parser.error(f"{name} must be >= 0")


def git_commit_or_none(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def optional_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_wheel_controller_facts(
    robot_description: str,
    wheel_names: list[str],
) -> tuple[dict[str, dict[str, Optional[float]]], bool, list[str]]:
    """Extract the controller input actually loaded by ros2_control from the expanded URDF."""
    errors: list[str] = []
    try:
        root = ET.fromstring(robot_description)
    except ET.ParseError as exc:
        return {}, False, [f"robot_description XML parse failed: {exc}"]

    ros2_control = root.find("./ros2_control[@name='MujocoSystem']")
    if ros2_control is None:
        ros2_control = root.find("./ros2_control")
    if ros2_control is None:
        return {}, False, ["robot_description has no ros2_control element"]

    by_joint: dict[str, dict[str, Optional[float]]] = {}
    pid_keys = {
        "kp": "velocity_kp",
        "ki": "velocity_ki",
        "kd": "velocity_kd",
        "i_min": "velocity_i_min",
        "i_max": "velocity_i_max",
    }
    for wheel_name in wheel_names:
        control_joint = ros2_control.find(f"./joint[@name='{wheel_name}']")
        urdf_joint = root.find(f"./joint[@name='{wheel_name}']")
        if control_joint is None:
            errors.append(f"missing ros2_control joint {wheel_name}")
            continue

        params = {
            str(param.get("name")): (param.text or "").strip()
            for param in control_joint.findall("./param")
            if param.get("name")
        }
        command_interface = control_joint.find("./command_interface[@name='velocity_pid']")
        limit = urdf_joint.find("./limit") if urdf_joint is not None else None
        facts: dict[str, Optional[float]] = {
            output_key: optional_float(params.get(parameter_name))
            for output_key, parameter_name in pid_keys.items()
        }
        facts.update(
            {
                "command_min_rad_s": optional_float(
                    command_interface.get("min") if command_interface is not None else None
                ),
                "command_max_rad_s": optional_float(
                    command_interface.get("max") if command_interface is not None else None
                ),
                "effort_limit_nm": optional_float(
                    limit.get("effort") if limit is not None else None
                ),
            }
        )
        missing = [key for key, value in facts.items() if value is None]
        if missing:
            errors.append(f"{wheel_name} missing values: {', '.join(missing)}")
        by_joint[wheel_name] = facts

    configurations = list(by_joint.values())
    uniform = bool(configurations) and len(configurations) == len(wheel_names)
    if uniform:
        uniform = all(configuration == configurations[0] for configuration in configurations[1:])
    return by_joint, uniform, errors


class FrequencyResponseRecorder(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("frequency_response_recorder")

        self.args = args
        self.cmd_topic = args.cmd_topic
        self.odom_topic = args.odom_topic
        self.dof_topic = args.dof_topic
        self.contact_topic = args.contact_topic

        created_at = datetime.now().astimezone()
        timestamp = created_at.strftime("%Y%m%d_%H%M%S")
        experiment_name = (
            f"{timestamp}_{sanitize_token(args.environment)}_{args.axis}_"
            f"A{format_float_token(float(args.amplitude), precision=4)}"
        )
        self.output_dir = Path(args.output_dir) / experiment_name
        self.runs_dir = self.output_dir / "runs"
        self.plots_dir = self.output_dir / "plots"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.output_dir / "metadata.json"

        self.created_at_wall = created_at.isoformat(timespec="seconds")
        self.finished = False
        self.started = False
        self.stop_reason: Optional[str] = None
        self.start_wall_time: Optional[str] = None
        self.end_wall_time: Optional[str] = None
        self.start_wall_monotonic: Optional[float] = None
        self.end_wall_monotonic: Optional[float] = None
        self.start_ros_time: Optional[float] = None
        self.end_ros_time: Optional[float] = None

        self.latest_odom: Optional[dict[str, float]] = None
        self.latest_odom_msg: Optional[Odometry] = None
        self.last_odom_wall_monotonic: Optional[float] = None
        self.received_odom = False
        self.received_dof = False
        self.received_contact = False
        self.next_wait_log_wall = time.monotonic()

        self.x0: Optional[float] = None
        self.y0: Optional[float] = None
        self.yaw0: Optional[float] = None

        self.frequency_index = -1
        self.current_run: Optional[RunContext] = None
        self.completed_runs: list[dict[str, Any]] = []

        self.odom_file: Optional[Any] = None
        self.cmd_file: Optional[Any] = None
        self.wheel_qfrc_file: Optional[Any] = None
        self.wheel_velocity_pid_file: Optional[Any] = None
        self.dof_file: Optional[Any] = None
        self.dof_raw_file: Optional[Any] = None
        self.contact_summary_file: Optional[Any] = None
        self.contact_file: Optional[Any] = None
        self.safety_file: Optional[Any] = None
        self.odom_writer: Optional[csv.writer] = None
        self.cmd_writer: Optional[csv.writer] = None
        self.wheel_qfrc_writer: Optional[csv.writer] = None
        self.wheel_velocity_pid_writer: Optional[csv.DictWriter] = None
        self.dof_writer: Optional[csv.writer] = None
        self.dof_raw_writer: Optional[csv.writer] = None
        self.contact_summary_writer: Optional[csv.writer] = None
        self.contact_writer: Optional[csv.writer] = None
        self.safety_writer: Optional[csv.writer] = None

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

        self.runtime_controller_facts = self._capture_runtime_controller_facts()
        capture_status = self.runtime_controller_facts["runtime_snapshot"]["status"]
        if args.require_runtime_metadata and capture_status != "complete":
            errors = self.runtime_controller_facts["runtime_snapshot"]["errors"]
            raise RuntimeError(
                "runtime controller metadata is incomplete: " + "; ".join(errors)
            )

        self._write_global_metadata()
        self.timer = self.create_timer(1.0 / float(args.publish_rate), self._on_timer)
        self.get_logger().info(
            f"Runtime metadata status={capture_status}. Waiting for first odometry sample on "
            f"{self.odom_topic}. Output: {self.output_dir}"
        )

    @staticmethod
    def _parameter_service_name(node_name: str) -> str:
        normalized = "/" + node_name.strip("/")
        return f"{normalized}/get_parameters"

    def _get_remote_parameters(
        self,
        node_name: str,
        names: list[str],
    ) -> tuple[dict[str, Any], list[str]]:
        service_name = self._parameter_service_name(node_name)
        client = self.create_client(GetParameters, service_name)
        errors: list[str] = []
        try:
            if not client.wait_for_service(timeout_sec=self.args.runtime_metadata_timeout):
                return {}, [f"parameter service unavailable: {service_name}"]

            request = GetParameters.Request()
            request.names = names
            future = client.call_async(request)
            rclpy.spin_until_future_complete(
                self,
                future,
                timeout_sec=self.args.runtime_metadata_timeout,
            )
            if not future.done():
                return {}, [f"parameter request timed out: {service_name}"]
            exception = future.exception()
            if exception is not None:
                return {}, [f"parameter request failed for {service_name}: {exception}"]

            response = future.result()
            if response is None or len(response.values) != len(names):
                return {}, [f"unexpected parameter response from {service_name}"]

            values: dict[str, Any] = {}
            for name, parameter_value in zip(names, response.values):
                value = parameter_value_to_python(parameter_value)
                values[name] = value
                if value is None:
                    errors.append(f"parameter not set: {node_name}:{name}")
            return values, errors
        finally:
            self.destroy_client(client)

    def _capture_runtime_controller_facts(self) -> dict[str, Any]:
        captured_at = datetime.now().astimezone().isoformat(timespec="seconds")
        errors: list[str] = []

        robot_description_values, query_errors = self._get_remote_parameters(
            self.args.robot_description_node,
            ["robot_description"],
        )
        errors.extend(query_errors)
        robot_description = robot_description_values.get("robot_description")
        if not isinstance(robot_description, str) or not robot_description.strip():
            robot_description = ""
            if not query_errors:
                errors.append("robot_description is empty")

        controller_manager, query_errors = self._get_remote_parameters(
            self.args.controller_manager_node,
            ["update_rate"],
        )
        errors.extend(query_errors)
        base_controller, query_errors = self._get_remote_parameters(
            self.args.base_controller_node,
            [
                "use_stamped_vel",
                "reference_timeout",
                "kinematics.wheels_radius",
                "kinematics.sum_of_robot_center_projection_on_X_Y_axis",
            ],
        )
        errors.extend(query_errors)
        sim_node, query_errors = self._get_remote_parameters(
            self.args.sim_node,
            [
                "base_ground_truth.publish_rate_hz",
                "model_package",
                "model_file",
            ],
        )
        errors.extend(query_errors)

        wheel_pid_by_joint: dict[str, dict[str, Optional[float]]] = {}
        wheel_pid_uniform = False
        if robot_description:
            wheel_pid_by_joint, wheel_pid_uniform, parse_errors = parse_wheel_controller_facts(
                robot_description,
                WHEEL_DOF_NAMES,
            )
            errors.extend(parse_errors)

        common_pid: Optional[dict[str, Optional[float]]] = None
        if wheel_pid_uniform and wheel_pid_by_joint:
            common_pid = copy.deepcopy(next(iter(wheel_pid_by_joint.values())))

        snapshot_status = "complete" if not errors else "partial"
        robot_description_sha256 = (
            hashlib.sha256(robot_description.encode("utf-8")).hexdigest()
            if robot_description
            else None
        )
        return {
            "runtime_snapshot": {
                "status": snapshot_status,
                "captured_at": captured_at,
                "robot_description_sha256": robot_description_sha256,
                "sources": {
                    "robot_description": (
                        f"{self.args.robot_description_node}:robot_description"
                    ),
                    "controller_manager": self.args.controller_manager_node,
                    "base_controller": self.args.base_controller_node,
                    "sim_node": self.args.sim_node,
                },
                "errors": errors,
            },
            "controller_manager_update_rate_hz": controller_manager.get("update_rate"),
            "base_ground_truth_publish_rate_hz": sim_node.get(
                "base_ground_truth.publish_rate_hz"
            ),
            "mecanum_base_controller_use_stamped_vel": base_controller.get(
                "use_stamped_vel"
            ),
            "mecanum_base_controller_reference_timeout_sec": base_controller.get(
                "reference_timeout"
            ),
            "wheel_radius_m": base_controller.get("kinematics.wheels_radius"),
            "lx_plus_ly_m": base_controller.get(
                "kinematics.sum_of_robot_center_projection_on_X_Y_axis"
            ),
            "mujoco_model": {
                "package": sim_node.get("model_package"),
                "file": sim_node.get("model_file"),
            },
            "wheel_velocity_pid": common_pid,
            "wheel_velocity_pid_uniform": wheel_pid_uniform,
            "wheel_velocity_pid_by_joint": wheel_pid_by_joint,
            "wheel_dof_names": list(WHEEL_DOF_NAMES),
        }

    def _on_odom(self, msg: Odometry) -> None:
        self.received_odom = True
        self.latest_odom_msg = msg
        self.last_odom_wall_monotonic = time.monotonic()

        pose = msg.pose.pose
        twist = msg.twist.twist
        yaw = yaw_from_quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        self.latest_odom = {
            "t_ros": stamp_to_sec(msg.header.stamp),
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
            "yaw": float(yaw),
            "vx": float(twist.linear.x),
            "vy": float(twist.linear.y),
            "wz": float(twist.angular.z),
        }

        if self._run_is_active():
            self._record_odom_sample()

    def _on_mujoco_dof_state(self, msg: MujocoDofState) -> None:
        self.received_dof = True
        if not self._run_is_active() or self.current_run is None:
            return

        t_ros = stamp_to_sec(msg.header.stamp)
        t_experiment = self._run_time_from_ros(t_ros)

        if (
            self.args.write_velocity_pid_diagnostics
            or self.args.require_pid_nonbinding_evidence
        ):
            try:
                pid_diagnostics = decode_velocity_pid_diagnostics(msg)
            except PidDiagnosticContractError as exc:
                self.current_run.velocity_pid_contract_errors += 1
                if self.current_run.velocity_pid_contract_errors == 1:
                    self.get_logger().error(
                        f"Velocity PID diagnostic contract error: {exc}"
                    )
            else:
                assert self.current_run.pid_nonbinding_evidence is not None
                try:
                    self.current_run.pid_nonbinding_evidence.update(pid_diagnostics)
                except PidDiagnosticContractError as exc:
                    self.current_run.velocity_pid_contract_errors += 1
                    if self.current_run.velocity_pid_contract_errors == 1:
                        self.get_logger().error(
                            f"Velocity PID nonbinding evidence error: {exc}"
                        )
                if self.wheel_velocity_pid_writer is not None:
                    for joint_name, diagnostic in pid_diagnostics.items():
                        self.wheel_velocity_pid_writer.writerow(
                            {
                                "t_experiment": f"{t_experiment:.9f}",
                                "header_stamp_s": f"{t_ros:.9f}",
                                "joint_name": joint_name,
                                **diagnostic,
                            }
                        )
                        self.current_run.wheel_velocity_pid_rows += 1

        names = list(msg.dof_names)
        applied = list(msg.qfrc_applied)
        dof_count = max(len(names), len(applied))

        wheel_qfrc: dict[str, float] = {}
        for i in range(dof_count):
            dof_name = names[i] if i < len(names) else f"dof_{i}"
            if dof_name in WHEEL_DOF_NAMES or "wheel" in dof_name:
                qfrc_applied = array_value(applied, i)
                try:
                    qfrc_value = float(qfrc_applied)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(qfrc_value):
                    wheel_qfrc[dof_name] = qfrc_value

        if self.wheel_qfrc_writer is not None:
            max_abs = math.nan
            max_wheel = ""
            if wheel_qfrc:
                max_wheel, max_value = max(
                    wheel_qfrc.items(),
                    key=lambda item: abs(item[1]),
                )
                max_abs = abs(max_value)
            self.wheel_qfrc_writer.writerow(
                [
                    f"{t_ros:.9f}",
                    f"{t_experiment:.9f}",
                    msg.header.frame_id,
                    len(wheel_qfrc),
                    max_abs,
                    max_wheel,
                ]
                + [wheel_qfrc.get(name, math.nan) for name in WHEEL_DOF_NAMES]
                + [int(msg.nefc), int(msg.ncon)]
            )
            self.current_run.wheel_qfrc_summary_rows += 1

        if self.dof_raw_writer is not None:
            self.dof_raw_writer.writerow(
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

        if self.dof_writer is not None:
            qpos = list(msg.qpos)
            qvel = list(msg.qvel)
            qacc = list(msg.qacc)
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
                dof_count,
                len(qpos),
                len(qvel),
                len(qacc),
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
                self.dof_writer.writerow(
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
        self.current_run.dof_samples += 1

    def _on_mujoco_contact_state(self, msg: MujocoContactState) -> None:
        self.received_contact = True
        if not self._run_is_active() or self.current_run is None:
            return

        t_ros = stamp_to_sec(msg.header.stamp)
        t_experiment = self._run_time_from_ros(t_ros)
        ncon = int(msg.ncon)
        contact_count = 0
        sum_force_norm_sq = 0.0
        max_force_norm = math.nan
        normal_abs_sum = 0.0
        tangent_norm_sum = 0.0

        for i in range(ncon):
            force = array_slice(msg.force_local_6d, 6 * i, 6)
            fx = float(force[0])
            fy = float(force[1])
            fz = float(force[2])
            if all(math.isfinite(value) for value in (fx, fy, fz)):
                force_norm_sq = fx * fx + fy * fy + fz * fz
                force_norm = math.sqrt(force_norm_sq)
                contact_count += 1
                sum_force_norm_sq += force_norm_sq
                max_force_norm = (
                    force_norm
                    if not math.isfinite(max_force_norm)
                    else max(max_force_norm, force_norm)
                )
                normal_abs_sum += abs(fx)
                tangent_norm_sum += math.hypot(fy, fz)

            if self.contact_writer is not None:
                position = array_slice(msg.position_xyz, 3 * i, 3)
                frame = array_slice(msg.frame_3x3, 9 * i, 9)
                friction = array_slice(msg.friction, 5 * i, 5)
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
                self.current_run.contact_rows += 1
        if self.contact_summary_writer is not None:
            self.contact_summary_writer.writerow(
                [
                    f"{t_ros:.9f}",
                    f"{t_experiment:.9f}",
                    msg.header.frame_id,
                    ncon,
                    contact_count,
                    sum_force_norm_sq,
                    max_force_norm,
                    normal_abs_sum,
                    tangent_norm_sum,
                ]
            )
            self.current_run.contact_force_summary_rows += 1
        self.current_run.contact_messages += 1

    def _on_timer(self) -> None:
        if self.finished:
            return

        if not self.received_odom:
            self._log_waiting_if_needed()
            return

        if not self.started:
            self._start_experiment()

        if self.current_run is None:
            self._prepare_next_run()
            return

        if not self.current_run.active:
            self._publish_zero_command(record=False)
            assert self.current_run.pre_settle_start_wall is not None
            if (
                time.monotonic() - self.current_run.pre_settle_start_wall
                >= float(self.args.settle_between_frequencies)
            ):
                self._start_current_run()
            return

        if self._odom_timed_out():
            self._record_safety_event("odom_timeout", "odom timeout before command update")
            self._complete_current_run(valid=False, reject_reason="odom_timeout")
            return

        safety_reason = self._safety_violation()
        if safety_reason is not None:
            self._record_safety_event(safety_reason, "safety boundary exceeded")
            self._complete_current_run(valid=False, reject_reason=safety_reason)
            return

        t_run = self._current_run_time()
        if t_run >= self.current_run.planned_duration:
            self._complete_current_run(valid=True, reject_reason="")
            return
        if t_run >= self.current_run.max_duration:
            self._record_safety_event("run_timeout", "run exceeded maximum duration")
            self._complete_current_run(valid=False, reject_reason="run_timeout")
            return

        self._publish_sine_command(t_run)

    def _start_experiment(self) -> None:
        if self.latest_odom is None:
            return
        self.started = True
        self.start_wall_time = datetime.now().astimezone().isoformat(timespec="seconds")
        self.start_wall_monotonic = time.monotonic()
        self.start_ros_time = self.latest_odom["t_ros"]
        self.x0 = self.latest_odom["x"]
        self.y0 = self.latest_odom["y"]
        self.yaw0 = self.latest_odom["yaw"]
        self._write_global_metadata()
        self.get_logger().info(
            f"Starting frequency sweep at x0={self.x0:.3f}, y0={self.y0:.3f}, "
            f"yaw0={self.yaw0:.3f} rad"
        )

    def _prepare_next_run(self) -> None:
        self.frequency_index += 1
        if self.frequency_index >= len(self.args.frequencies):
            self._finish("completed")
            return

        frequency = float(self.args.frequencies[self.frequency_index])
        run_id = format_frequency_run_id(frequency)
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        period = 1.0 / frequency
        planned_duration = float(self.args.cycles) * period
        max_duration = planned_duration + 2.0
        self.current_run = RunContext(
            run_id=run_id,
            frequency_hz=frequency,
            omega_rad_s=2.0 * math.pi * frequency,
            run_dir=run_dir,
            planned_duration=planned_duration,
            max_duration=max_duration,
            pre_settle_start_wall=time.monotonic(),
            pid_nonbinding_evidence=PidNonbindingEvidence(frequency),
        )
        self._open_run_files(run_dir)
        self._write_run_metadata()
        self.get_logger().info(
            f"Prepared {run_id}: frequency={frequency:.3f} Hz, "
            f"duration={planned_duration:.3f} s"
        )

    def _start_current_run(self) -> None:
        if self.current_run is None or self.latest_odom is None:
            return

        run = self.current_run
        run.active = True
        run.start_wall_time = datetime.now().astimezone().isoformat(timespec="seconds")
        run.start_wall_monotonic = time.monotonic()
        run.start_ros_time = self.latest_odom["t_ros"]
        run.x_start = self.latest_odom["x"]
        run.y_start = self.latest_odom["y"]
        run.yaw_start = self.latest_odom["yaw"]
        self._write_run_metadata()
        self._record_odom_sample()
        self.get_logger().info(f"Running {run.run_id}")

    def _complete_current_run(self, valid: bool, reject_reason: str) -> None:
        if self.current_run is None:
            return

        run = self.current_run
        if self.args.require_pid_nonbinding_evidence:
            evidence = (
                run.pid_nonbinding_evidence.to_dict()
                if run.pid_nonbinding_evidence is not None
                else {"complete": False}
            )
            if run.velocity_pid_contract_errors:
                valid = False
                reject_reason = "pid_nonbinding_evidence_contract_error"
            elif not evidence.get("complete", False):
                valid = False
                reject_reason = "pid_nonbinding_evidence_missing"
        run.active = False
        run.completed = True
        run.valid = bool(valid)
        run.reject_reason = reject_reason
        run.end_wall_time = datetime.now().astimezone().isoformat(timespec="seconds")
        run.end_wall_monotonic = time.monotonic()
        run.end_ros_time = (
            self.latest_odom["t_ros"]
            if self.latest_odom is not None
            else self._node_now_sec()
        )
        if self.latest_odom is not None:
            run.x_final = self.latest_odom["x"]
            run.y_final = self.latest_odom["y"]
            run.yaw_final = self.latest_odom["yaw"]

        self._publish_zero_commands(duration_sec=0.25)
        self._write_run_metadata()
        self.completed_runs.append(self._run_metadata_dict(run))
        self._close_run_files()
        self.get_logger().info(
            f"Completed {run.run_id}: valid={run.valid}, reject_reason='{run.reject_reason}'"
        )
        self.current_run = None
        self._write_global_metadata()

    def _run_is_active(self) -> bool:
        return self.current_run is not None and self.current_run.active and not self.finished

    def _record_odom_sample(self) -> None:
        if self.latest_odom is None or self.current_run is None or self.odom_writer is None:
            return
        t_ros = self.latest_odom["t_ros"]
        t_experiment = self._run_time_from_ros(t_ros)
        self.odom_writer.writerow(
            [
                f"{t_ros:.9f}",
                f"{t_experiment:.9f}",
                self.latest_odom["x"],
                self.latest_odom["y"],
                self.latest_odom["z"],
                self.latest_odom["yaw"],
                self.latest_odom["vx"],
                self.latest_odom["vy"],
                self.latest_odom["wz"],
            ]
        )
        self.current_run.odom_samples += 1

    def _publish_sine_command(self, t_run: float) -> None:
        if self.current_run is None:
            return
        phase = self.current_run.omega_rad_s * t_run
        envelope = self._envelope(t_run, self.current_run.frequency_hz)
        command = float(self.args.amplitude) * envelope * math.sin(phase)
        vx = command if self.args.axis == "x" else 0.0
        vy = 0.0
        wz = 0.0
        self._publish_command(
            vx,
            vy,
            wz,
            t_experiment=t_run,
            frequency_hz=self.current_run.frequency_hz,
            omega_rad_s=self.current_run.omega_rad_s,
            phase_rad=phase,
            envelope=envelope,
            record=True,
        )

    def _publish_zero_command(self, record: bool = False) -> None:
        t_run = self._current_run_time()
        frequency_hz = self.current_run.frequency_hz if self.current_run is not None else 0.0
        omega = self.current_run.omega_rad_s if self.current_run is not None else 0.0
        self._publish_command(
            0.0,
            0.0,
            0.0,
            t_experiment=t_run,
            frequency_hz=frequency_hz,
            omega_rad_s=omega,
            phase_rad=0.0,
            envelope=0.0,
            record=record,
        )

    def _publish_zero_commands(
        self,
        duration_sec: float = 0.25,
        interval_sec: float = 0.025,
    ) -> None:
        end_time = time.monotonic() + max(0.0, duration_sec)
        while time.monotonic() < end_time:
            if not rclpy.ok():
                break
            self._publish_zero_command(record=False)
            time.sleep(interval_sec)

    def _publish_command(
        self,
        vx: float,
        vy: float,
        wz: float,
        t_experiment: float,
        frequency_hz: float,
        omega_rad_s: float,
        phase_rad: float,
        envelope: float,
        record: bool,
    ) -> None:
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)

        if record and self.current_run is not None and self.cmd_writer is not None:
            t_ros = (
                self.latest_odom["t_ros"]
                if self.latest_odom is not None
                else self._node_now_sec()
            )
            self.cmd_writer.writerow(
                [
                    f"{t_ros:.9f}",
                    f"{t_experiment:.9f}",
                    f"{frequency_hz:.9f}",
                    f"{omega_rad_s:.9f}",
                    self.args.axis,
                    f"{phase_rad:.9f}",
                    f"{envelope:.9f}",
                    vx,
                    vy,
                    wz,
                ]
            )
            self.current_run.cmd_samples += 1

    def _envelope(self, t_run: float, frequency_hz: float) -> float:
        ramp_cycles = float(self.args.ramp_cycles)
        if ramp_cycles <= 0.0:
            return 1.0
        ramp_duration = ramp_cycles / frequency_hz
        if ramp_duration <= 1e-9:
            return 1.0
        ratio = max(0.0, min(1.0, t_run / ramp_duration))
        return 0.5 - 0.5 * math.cos(math.pi * ratio)

    def _odom_timed_out(self) -> bool:
        if self.last_odom_wall_monotonic is None:
            return False
        return time.monotonic() - self.last_odom_wall_monotonic > float(self.args.odom_timeout)

    def _safety_violation(self) -> Optional[str]:
        if self.latest_odom is None or self.x0 is None or self.y0 is None or self.yaw0 is None:
            return None
        if abs(self.latest_odom["x"] - self.x0) > float(self.args.max_x):
            return "max_x"
        if abs(self.latest_odom["y"] - self.y0) > float(self.args.max_y):
            return "max_y"
        max_yaw = math.radians(float(self.args.max_yaw_deg))
        if abs(wrap_angle(self.latest_odom["yaw"] - self.yaw0)) > max_yaw:
            return "max_yaw"
        return None

    def _record_safety_event(self, event: str, detail: str) -> None:
        if self.current_run is None or self.safety_writer is None:
            return
        t_ros = self.latest_odom["t_ros"] if self.latest_odom is not None else self._node_now_sec()
        t_experiment = self._current_run_time()
        x = self.latest_odom["x"] if self.latest_odom is not None else math.nan
        y = self.latest_odom["y"] if self.latest_odom is not None else math.nan
        yaw = self.latest_odom["yaw"] if self.latest_odom is not None else math.nan
        self.safety_writer.writerow(
            [
                f"{t_ros:.9f}",
                f"{t_experiment:.9f}",
                event,
                detail,
                x,
                y,
                yaw,
            ]
        )

    def _run_time_from_ros(self, t_ros: float) -> float:
        if self.current_run is not None and self.current_run.start_ros_time is not None:
            return max(0.0, t_ros - self.current_run.start_ros_time)
        return 0.0

    def _current_run_time(self) -> float:
        if self.current_run is None:
            return 0.0
        if self.latest_odom is not None and self.current_run.start_ros_time is not None:
            return self._run_time_from_ros(self.latest_odom["t_ros"])
        if self.current_run.start_wall_monotonic is not None:
            return max(0.0, time.monotonic() - self.current_run.start_wall_monotonic)
        return 0.0

    def _node_now_sec(self) -> float:
        return stamp_to_sec(self.get_clock().now().to_msg())

    def _log_waiting_if_needed(self) -> None:
        now = time.monotonic()
        if now >= self.next_wait_log_wall:
            self.get_logger().info(f"Waiting for odometry on {self.odom_topic}...")
            self.next_wait_log_wall = now + 2.0

    def _open_run_files(self, run_dir: Path) -> None:
        self.odom_file = (run_dir / "odom.csv").open("w", newline="")
        self.cmd_file = (run_dir / "cmd_vel.csv").open("w", newline="")
        self.wheel_qfrc_file = (run_dir / "wheel_qfrc_summary.csv").open("w", newline="")
        self.wheel_velocity_pid_file = (
            (run_dir / "wheel_velocity_pid.csv").open("w", newline="")
            if self.args.write_velocity_pid_diagnostics
            else None
        )
        self.dof_file = (
            (run_dir / "mujoco_dof_state.csv").open("w", newline="")
            if self.args.write_full_dof_state
            else None
        )
        self.dof_raw_file = (
            (run_dir / "mujoco_dof_state_raw.csv").open("w", newline="")
            if self.args.write_raw_dof_state
            else None
        )
        self.contact_summary_file = (run_dir / "contact_force_summary.csv").open(
            "w", newline=""
        )
        self.contact_file = (
            (run_dir / "contact_forces.csv").open("w", newline="")
            if self.args.write_contact_forces
            else None
        )
        self.safety_file = (run_dir / "safety_events.csv").open("w", newline="")
        self.odom_writer = csv.writer(self.odom_file)
        self.cmd_writer = csv.writer(self.cmd_file)
        self.wheel_qfrc_writer = csv.writer(self.wheel_qfrc_file)
        self.wheel_velocity_pid_writer = (
            csv.DictWriter(self.wheel_velocity_pid_file, fieldnames=PID_CSV_FIELDS)
            if self.wheel_velocity_pid_file is not None
            else None
        )
        self.dof_writer = csv.writer(self.dof_file) if self.dof_file is not None else None
        self.dof_raw_writer = (
            csv.writer(self.dof_raw_file) if self.dof_raw_file is not None else None
        )
        self.contact_summary_writer = csv.writer(self.contact_summary_file)
        self.contact_writer = (
            csv.writer(self.contact_file) if self.contact_file is not None else None
        )
        self.safety_writer = csv.writer(self.safety_file)
        self._write_run_headers()

    def _write_run_headers(self) -> None:
        assert self.odom_writer is not None
        assert self.cmd_writer is not None
        assert self.wheel_qfrc_writer is not None
        assert self.contact_summary_writer is not None
        assert self.safety_writer is not None

        self.cmd_writer.writerow(
            [
                "t_ros",
                "t_experiment",
                "frequency_hz",
                "omega_rad_s",
                "axis",
                "phase_rad",
                "envelope",
                "cmd_vx",
                "cmd_vy",
                "cmd_wz",
            ]
        )
        self.odom_writer.writerow(
            ["t_ros", "t_experiment", "x", "y", "z", "yaw", "vx", "vy", "wz"]
        )
        self.wheel_qfrc_writer.writerow(WHEEL_QFRC_SUMMARY_FIELDS)
        if self.wheel_velocity_pid_writer is not None:
            self.wheel_velocity_pid_writer.writeheader()
        if self.dof_writer is not None:
            self.dof_writer.writerow(
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
        if self.dof_raw_writer is not None:
            self.dof_raw_writer.writerow(
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
        self.contact_summary_writer.writerow(CONTACT_FORCE_SUMMARY_FIELDS)
        if self.contact_writer is not None:
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
        self.safety_writer.writerow(
            ["t_ros", "t_experiment", "event", "detail", "x", "y", "yaw"]
        )

    def _close_run_files(self) -> None:
        for file_obj in (
            self.odom_file,
            self.cmd_file,
            self.wheel_qfrc_file,
            self.wheel_velocity_pid_file,
            self.dof_file,
            self.dof_raw_file,
            self.contact_summary_file,
            self.contact_file,
            self.safety_file,
        ):
            if file_obj is not None and not file_obj.closed:
                file_obj.close()
        self.odom_file = None
        self.cmd_file = None
        self.wheel_qfrc_file = None
        self.wheel_velocity_pid_file = None
        self.dof_file = None
        self.dof_raw_file = None
        self.contact_summary_file = None
        self.contact_file = None
        self.safety_file = None
        self.odom_writer = None
        self.cmd_writer = None
        self.wheel_qfrc_writer = None
        self.wheel_velocity_pid_writer = None
        self.dof_writer = None
        self.dof_raw_writer = None
        self.contact_summary_writer = None
        self.contact_writer = None
        self.safety_writer = None

    def _write_global_metadata(self) -> None:
        metadata = {
            "metadata_schema_version": 2,
            "experiment_type": "frequency_response",
            "created_at": self.created_at_wall,
            "git_commit": git_commit_or_none(Path.cwd()),
            "ros_distro": "humble",
            "mujoco_version": "3.3.0",
            "robot": "RB-KAIROS",
            "environment": self.args.environment,
            "axis": self.args.axis,
            "script_parameters": {
                "amplitude": self.args.amplitude,
                "frequencies": self.args.frequencies,
                "publish_rate": self.args.publish_rate,
                "cycles": self.args.cycles,
                "discard_cycles": self.args.discard_cycles,
                "ramp_cycles": self.args.ramp_cycles,
                "output_dir": self.args.output_dir,
                "max_x": self.args.max_x,
                "max_y": self.args.max_y,
                "max_yaw_deg": self.args.max_yaw_deg,
                "odom_timeout": self.args.odom_timeout,
                "settle_between_frequencies": self.args.settle_between_frequencies,
                "runtime_metadata_timeout": self.args.runtime_metadata_timeout,
                "require_runtime_metadata": self.args.require_runtime_metadata,
                "write_full_dof_state": self.args.write_full_dof_state,
                "write_raw_dof_state": self.args.write_raw_dof_state,
                "write_contact_forces": self.args.write_contact_forces,
            },
            "topics": self._topics_dict(),
            "controller_facts": self._controller_facts(),
            "frequency_list": self.args.frequencies,
            "output_dir": str(self.output_dir),
            "runs_dir": str(self.runs_dir),
            "plots_dir": str(self.plots_dir),
            "initial_state": {"x0": self.x0, "y0": self.y0, "yaw0": self.yaw0},
            "wall_time": {
                "start": self.start_wall_time,
                "end": self.end_wall_time,
            },
            "ros_time": {
                "start_ros_time": self.start_ros_time,
                "end_ros_time": self.end_ros_time,
                "per_run_t_experiment_source": (
                    "relative_to_run_start_base_ground_truth_odom_header_stamp"
                ),
            },
            "completed_runs": self.completed_runs,
            "stop_reason": self.stop_reason,
            "notes": [
                "Only x-axis simulation frequency response is implemented in this first version.",
                "cmd_vel.csv records the command actually published by this script.",
                "odom.csv uses base_ground_truth_odom twist.linear.x as the primary output.",
                (
                    "wheel_qfrc_summary.csv is the default lightweight wheel "
                    "qfrc_applied diagnostic used for saturation_ratio."
                ),
                (
                    "contact_force_summary.csv is the default lightweight contact "
                    "diagnostic used for contact_force_rms."
                ),
                (
                    "mujoco_dof_state.csv, mujoco_dof_state_raw.csv, and "
                    "contact_forces.csv are written only when their explicit debug "
                    "flags are set."
                ),
            ],
        }
        self._write_json(self.metadata_path, metadata)

    def _write_run_metadata(self) -> None:
        if self.current_run is None:
            return
        self._write_json(
            self.current_run.run_dir / "run_metadata.json",
            self._run_metadata_dict(self.current_run),
        )

    def _run_metadata_dict(self, run: RunContext) -> dict[str, Any]:
        period = 1.0 / run.frequency_hz
        theoretical_position_amplitude = float(self.args.amplitude) / run.omega_rad_s
        actual_duration = None
        if run.start_ros_time is not None and run.end_ros_time is not None:
            actual_duration = max(0.0, run.end_ros_time - run.start_ros_time)
        return {
            "metadata_schema_version": 2,
            "run_id": run.run_id,
            "environment": self.args.environment,
            "axis": self.args.axis,
            "frequency_hz": run.frequency_hz,
            "omega_rad_s": run.omega_rad_s,
            "period_sec": period,
            "amplitude": self.args.amplitude,
            "cycles": self.args.cycles,
            "discard_cycles": self.args.discard_cycles,
            "ramp_cycles": self.args.ramp_cycles,
            "publish_rate": self.args.publish_rate,
            "planned_duration": run.planned_duration,
            "max_duration": run.max_duration,
            "actual_duration": actual_duration,
            "valid": run.valid,
            "reject_reason": run.reject_reason,
            "active": run.active,
            "completed": run.completed,
            "x0": self.x0,
            "y0": self.y0,
            "yaw0": self.yaw0,
            "x_start": run.x_start,
            "y_start": run.y_start,
            "yaw_start": run.yaw_start,
            "x_final": run.x_final,
            "y_final": run.y_final,
            "yaw_final": run.yaw_final,
            "theoretical_position_amplitude": theoretical_position_amplitude,
            "displacement_risk": theoretical_position_amplitude > 0.35,
            "topics": self._topics_dict(),
            "controller_facts": self._controller_facts(),
            "wall_time": {
                "start": run.start_wall_time,
                "end": run.end_wall_time,
            },
            "ros_time": {
                "start_ros_time": run.start_ros_time,
                "end_ros_time": run.end_ros_time,
                "t_experiment_source": "relative_to_run_start_base_ground_truth_odom_header_stamp",
            },
            "samples": {
                "cmd": run.cmd_samples,
                "odom": run.odom_samples,
                "mujoco_dof_state_messages": run.dof_samples,
                "wheel_qfrc_summary_rows": run.wheel_qfrc_summary_rows,
                "wheel_velocity_pid_rows": run.wheel_velocity_pid_rows,
                "velocity_pid_contract_errors": run.velocity_pid_contract_errors,
                "mujoco_contact_state_messages": run.contact_messages,
                "contact_force_summary_rows": run.contact_force_summary_rows,
                "contact_rows": run.contact_rows,
            },
            "frf_pid_nonbinding_evidence": (
                run.pid_nonbinding_evidence.to_dict()
                if run.pid_nonbinding_evidence is not None
                and (
                    self.args.write_velocity_pid_diagnostics
                    or self.args.require_pid_nonbinding_evidence
                )
                else None
            ),
            "output_files": {
                "cmd_vel": str(run.run_dir / "cmd_vel.csv"),
                "odom": str(run.run_dir / "odom.csv"),
                "wheel_qfrc_summary": str(run.run_dir / "wheel_qfrc_summary.csv"),
                "wheel_velocity_pid": (
                    str(run.run_dir / "wheel_velocity_pid.csv")
                    if self.args.write_velocity_pid_diagnostics
                    else None
                ),
                "mujoco_dof_state": (
                    str(run.run_dir / "mujoco_dof_state.csv")
                    if self.args.write_full_dof_state
                    else None
                ),
                "mujoco_dof_state_raw": (
                    str(run.run_dir / "mujoco_dof_state_raw.csv")
                    if self.args.write_raw_dof_state
                    else None
                ),
                "contact_force_summary": str(run.run_dir / "contact_force_summary.csv"),
                "contact_forces": (
                    str(run.run_dir / "contact_forces.csv")
                    if self.args.write_contact_forces
                    else None
                ),
                "safety_events": str(run.run_dir / "safety_events.csv"),
                "run_metadata": str(run.run_dir / "run_metadata.json"),
            },
        }

    def _topics_dict(self) -> dict[str, str]:
        return {
            "cmd_vel": self.cmd_topic,
            "ground_truth_odom": self.odom_topic,
            "mujoco_dof_state": self.dof_topic,
            "mujoco_dof_state_type": "mujoco_ros2_control_msgs/msg/MujocoDofState",
            "mujoco_contact_state": self.contact_topic,
            "mujoco_contact_state_type": "mujoco_ros2_control_msgs/msg/MujocoContactState",
        }

    def _controller_facts(self) -> dict[str, Any]:
        return copy.deepcopy(self.runtime_controller_facts)

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, indent=2, sort_keys=True)
            file_obj.write("\n")

    def _finish(self, reason: str) -> None:
        if self.finished:
            return
        self.stop_reason = reason
        if self.current_run is not None:
            if self.current_run.active:
                self._complete_current_run(valid=False, reject_reason=reason)
            else:
                self._close_run_files()
                self.current_run = None
        self._publish_zero_commands(duration_sec=0.25)
        self.finished = True
        self.end_wall_time = datetime.now().astimezone().isoformat(timespec="seconds")
        self.end_wall_monotonic = time.monotonic()
        self.end_ros_time = self._node_now_sec()
        self._write_global_metadata()
        self.get_logger().info(f"Finished with stop_reason='{reason}'. Output: {self.output_dir}")

    def destroy_node(self) -> bool:
        self._close_run_files()
        return super().destroy_node()


def parse_args(argv: Optional[list[str]] = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Record closed-loop base velocity frequency response for RB-KAIROS MuJoCo."
    )
    parser.add_argument("--environment", default="sim", help="experiment environment label")
    parser.add_argument("--axis", default="x", choices=["x"], help="commanded base velocity axis")
    parser.add_argument("--amplitude", type=float, default=0.05, help="sine velocity amplitude")
    parser.add_argument(
        "--frequencies",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.50, 0.80, 1.20, 2.00],
        help="frequency list [Hz]",
    )
    parser.add_argument(
        "--publish-rate",
        type=float,
        default=100.0,
        help="command publish rate [Hz]",
    )
    parser.add_argument("--cycles", type=float, default=6.0, help="cycles per frequency")
    parser.add_argument(
        "--discard-cycles",
        type=float,
        default=2.0,
        help="cycles discarded by the analyzer from each run",
    )
    parser.add_argument("--ramp-cycles", type=float, default=1.0, help="smooth ramp-in cycles")
    parser.add_argument(
        "--output-dir",
        default="data/frequency_response",
        help="base output directory",
    )
    parser.add_argument("--cmd-topic", default=CMD_TOPIC_DEFAULT, help="Twist command topic")
    parser.add_argument("--odom-topic", default=ODOM_TOPIC_DEFAULT, help="ground-truth odom topic")
    parser.add_argument(
        "--dof-topic",
        default=DOF_TOPIC_DEFAULT,
        help="MujocoDofState topic containing qpos/qvel/qacc/qfrc data",
    )
    parser.add_argument(
        "--write-full-dof-state",
        action="store_true",
        help=(
            "also write full per-DOF mujoco_dof_state.csv; default keeps only "
            "wheel_qfrc_summary.csv"
        ),
    )
    parser.add_argument(
        "--write-raw-dof-state",
        action="store_true",
        help="also write raw-array mujoco_dof_state_raw.csv for debugging",
    )
    parser.add_argument(
        "--write-velocity-pid-diagnostics",
        action="store_true",
        help="write the versioned four-wheel PID diagnostic CSV; disabled by default",
    )
    parser.add_argument(
        "--require-pid-nonbinding-evidence",
        action="store_true",
        help=(
            "require versioned controller-rate PID clamp latches and candidate "
            "I-term peaks for conditional FRF reuse; no raw PID CSV is written"
        ),
    )
    parser.add_argument(
        "--contact-topic",
        default=CONTACT_TOPIC_DEFAULT,
        help="MujocoContactState topic containing per-contact force data",
    )
    parser.add_argument(
        "--write-contact-forces",
        action="store_true",
        help=(
            "also write full per-contact contact_forces.csv; default keeps only "
            "contact_force_summary.csv"
        ),
    )
    parser.add_argument(
        "--robot-description-node",
        default=ROBOT_DESCRIPTION_NODE_DEFAULT,
        help="node exposing the effective robot_description parameter",
    )
    parser.add_argument(
        "--controller-manager-node",
        default=CONTROLLER_MANAGER_NODE_DEFAULT,
        help="controller manager node used for the runtime metadata snapshot",
    )
    parser.add_argument(
        "--base-controller-node",
        default=BASE_CONTROLLER_NODE_DEFAULT,
        help="mecanum controller node used for the runtime metadata snapshot",
    )
    parser.add_argument(
        "--sim-node",
        default=SIM_NODE_DEFAULT,
        help="MuJoCo simulation node used for the runtime metadata snapshot",
    )
    parser.add_argument(
        "--runtime-metadata-timeout",
        type=float,
        default=2.0,
        help="timeout for each runtime parameter service request [s]",
    )
    parser.add_argument(
        "--require-runtime-metadata",
        action="store_true",
        help="refuse to start when the runtime controller snapshot is incomplete",
    )
    parser.add_argument(
        "--max-x", type=float, default=0.4,
        help="relative world x safety limit [m]",
    )
    parser.add_argument(
        "--max-y", type=float, default=0.15,
        help="relative world y safety limit [m]",
    )
    parser.add_argument(
        "--max-yaw-deg",
        type=float,
        default=10.0,
        help="relative yaw safety limit [deg]",
    )
    parser.add_argument(
        "--odom-timeout",
        type=float,
        default=1.0,
        help="maximum wall-clock time without odom before stopping a run [s]",
    )
    parser.add_argument(
        "--settle-between-frequencies",
        type=float,
        default=1.0,
        help="zero-command settle time before each frequency [s]",
    )

    args, ros_args = parser.parse_known_args(argv)
    if ros_args:
        if "--ros-args" not in ros_args:
            parser.error("unknown arguments: " + " ".join(ros_args))
        ros_arg_index = ros_args.index("--ros-args")
        if ros_arg_index > 0:
            parser.error("unknown arguments: " + " ".join(ros_args[:ros_arg_index]))
        ros_args = ros_args[ros_arg_index:]

    if not args.environment.strip():
        parser.error("--environment must not be empty")
    positive_float(parser, "--amplitude", args.amplitude)
    if not args.frequencies:
        parser.error("--frequencies must contain at least one value")
    for frequency in args.frequencies:
        positive_float(parser, "--frequencies", frequency)
    positive_float(parser, "--publish-rate", args.publish_rate)
    positive_float(parser, "--cycles", args.cycles)
    nonnegative_float(parser, "--discard-cycles", args.discard_cycles)
    nonnegative_float(parser, "--ramp-cycles", args.ramp_cycles)
    if args.discard_cycles >= args.cycles:
        parser.error("--discard-cycles must be smaller than --cycles")
    positive_float(parser, "--max-x", args.max_x)
    positive_float(parser, "--max-y", args.max_y)
    positive_float(parser, "--max-yaw-deg", args.max_yaw_deg)
    positive_float(parser, "--odom-timeout", args.odom_timeout)
    positive_float(parser, "--runtime-metadata-timeout", args.runtime_metadata_timeout)
    nonnegative_float(parser, "--settle-between-frequencies", args.settle_between_frequencies)

    return args, ros_args


def main(argv: Optional[list[str]] = None) -> None:
    args, ros_args = parse_args(argv)
    rclpy.init(args=ros_args)
    node = FrequencyResponseRecorder(args)
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
