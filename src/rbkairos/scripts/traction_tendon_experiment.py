#!/usr/bin/env python3
"""Record an adaptive steady-state MuJoCo limited-tendon traction experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import mujoco
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist, WrenchStamped
from mujoco_ros2_control_msgs.msg import MujocoContactState, MujocoDofState
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray

from traction_tendon_analysis import (
    AXIS_CONTRACTS,
    CORE_SUMMARY_FIELDS,
    EXPERIMENT_TYPE,
    FORCE_SIGN_CONVENTION,
    MEASUREMENT_METHOD,
    STATE_FIELDS,
    STATE_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    AnalysisInputError,
    analyze_traction_tendon_run,
    decode_state_message,
)
from traction_tendon_steady import (
    STEADY_METHOD,
    AdaptiveSteadyMonitor,
    SteadyStateConfig,
)


CMD_TOPIC_DEFAULT = "/mecanum_base_controller/reference_unstamped"
ODOM_TOPIC_DEFAULT = "/base_ground_truth_odom"
STATE_TOPIC_DEFAULT = "/traction_tendon/state"
WRENCH_TOPIC_DEFAULT = "/traction_tendon/wrench"
DOF_TOPIC_DEFAULT = "/mujoco_dof_state"
CONTACT_TOPIC_DEFAULT = "/mujoco_contact_state"
MUJOCO_NODE_DEFAULT = "/mujoco_sim_ros2_node"
WHEEL_DOF_ORDER = (
    "left_front_wheel_joint",
    "right_front_wheel_joint",
    "left_back_wheel_joint",
    "right_back_wheel_joint",
)
PID_DIAGNOSTIC_SCHEMA_VERSION = 2
PID_RUNTIME_METADATA_SCHEMA_VERSION = 1
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
UR5_JOINT_ORDER = (
    "ur5/shoulder_pan_joint",
    "ur5/shoulder_lift_joint",
    "ur5/elbow_joint",
    "ur5/wrist_1_joint",
    "ur5/wrist_2_joint",
    "ur5/wrist_3_joint",
)
UR5_PRESETS = {
    "up": [0.0, -math.pi / 2.0, 0.0, -math.pi / 2.0, 0.0, 0.0],
}
RUNTIME_PARAMETER_NAMES = (
    "model_file",
    "physics_plugins",
    "traction_tendon.tendon_name",
    "traction_tendon.sensor_name",
    "traction_tendon.attach_site_name",
    "traction_tendon.anchor_site_name",
    "traction_tendon.publish_rate_hz",
    "traction_tendon.state_topic",
    "traction_tendon.wrench_topic",
)


class PidDiagnosticContractError(RuntimeError):
    """Raised when the versioned velocity PID diagnostic block is inconsistent."""


def decode_ur5_state(message: Any) -> dict[str, list[float]]:
    names = list(getattr(message, "dof_names", []))
    positions = list(getattr(message, "qpos", []))
    velocities = list(getattr(message, "qvel", []))
    if len(positions) != len(names) or len(velocities) != len(names):
        raise PidDiagnosticContractError("UR5 qpos/qvel length does not match dof_names")
    index = {name: offset for offset, name in enumerate(names)}
    if any(name not in index for name in UR5_JOINT_ORDER):
        raise PidDiagnosticContractError("MuJoCo DOF state is missing UR5 joints")
    result = {
        "positions_rad": [float(positions[index[name]]) for name in UR5_JOINT_ORDER],
        "velocities_rad_s": [float(velocities[index[name]]) for name in UR5_JOINT_ORDER],
    }
    if not all(math.isfinite(value) for values in result.values() for value in values):
        raise PidDiagnosticContractError("UR5 state contains non-finite values")
    return result


def validate_ur5_pose(
    state: dict[str, list[float]],
    *,
    preset: str,
    position_tolerance_rad: float,
    velocity_tolerance_rad_s: float,
) -> Optional[str]:
    target = UR5_PRESETS.get(preset)
    if target is None:
        return f"unknown required UR5 preset {preset!r}"
    errors = [
        abs(measured - expected)
        for measured, expected in zip(state["positions_rad"], target)
    ]
    speeds = [abs(value) for value in state["velocities_rad_s"]]
    if max(errors, default=math.inf) > position_tolerance_rad:
        return (
            f"UR5 preset {preset} position error {max(errors):.6g} rad exceeds "
            f"{position_tolerance_rad:.6g} rad"
        )
    if max(speeds, default=math.inf) > velocity_tolerance_rad_s:
        return (
            f"UR5 preset {preset} speed {max(speeds):.6g} rad/s exceeds "
            f"{velocity_tolerance_rad_s:.6g} rad/s"
        )
    return None


def decode_velocity_pid_diagnostics(message: Any) -> dict[str, dict[str, Any]]:
    version = int(getattr(message, "velocity_pid_diagnostic_schema_version", -1))
    if version != PID_DIAGNOSTIC_SCHEMA_VERSION:
        raise PidDiagnosticContractError(
            f"velocity PID diagnostic schema {version} != {PID_DIAGNOSTIC_SCHEMA_VERSION}"
        )

    names = list(getattr(message, "velocity_pid_joint_names", []))
    if len(names) != len(set(names)) or set(names) != set(WHEEL_DOF_ORDER):
        raise PidDiagnosticContractError(
            "velocity PID diagnostics must contain exactly the four wheel joints"
        )
    count = len(names)
    decoded_by_name: dict[str, dict[str, Any]] = {name: {} for name in names}

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
            decoded_by_name[name][record_field] = value

    for message_field, record_field in PID_BOOL_FIELDS:
        values = list(getattr(message, message_field, []))
        if len(values) != count:
            raise PidDiagnosticContractError(
                f"{message_field} has {len(values)} values, expected {count}"
            )
        for name, value in zip(names, values):
            decoded_by_name[name][record_field] = bool(value)

    for message_field, record_field in PID_INTEGER_FIELDS:
        values = list(getattr(message, message_field, []))
        if len(values) != count:
            raise PidDiagnosticContractError(
                f"{message_field} has {len(values)} values, expected {count}"
            )
        for name, value in zip(names, values):
            parsed = int(value)
            if parsed <= 0:
                raise PidDiagnosticContractError(
                    f"invalid {message_field} for {name}"
                )
            decoded_by_name[name][record_field] = parsed

    dof_names = list(getattr(message, "dof_names", []))
    qfrc_applied = list(getattr(message, "qfrc_applied", []))
    if len(qfrc_applied) != len(dof_names):
        raise PidDiagnosticContractError("qfrc_applied length does not match dof_names")
    dof_index = {name: index for index, name in enumerate(dof_names)}

    for name, record in decoded_by_name.items():
        if name not in dof_index:
            raise PidDiagnosticContractError(f"PID joint {name} is absent from dof_names")
        qfrc = float(qfrc_applied[dof_index[name]])
        if not math.isfinite(qfrc):
            raise PidDiagnosticContractError(f"non-finite qfrc_applied for {name}")
        record["qfrc_applied"] = qfrc

        if not math.isclose(
            record["velocity_error"],
            record["velocity_target"] - record["velocity_actual"],
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise PidDiagnosticContractError(f"velocity error identity failed for {name}")
        if not math.isclose(
            record["p_term"],
            record["kp"] * record["velocity_error"],
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise PidDiagnosticContractError(f"P term identity failed for {name}")
        if not math.isclose(
            record["p_term"] + record["i_term"] + record["d_term"],
            record["pid_raw_output"],
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise PidDiagnosticContractError(f"P+I+D identity failed for {name}")
        if record["i_min"] > record["i_max"]:
            raise PidDiagnosticContractError(f"invalid integral limits for {name}")
        if not (
            record["i_min"] - 1e-9
            <= record["i_term"]
            <= record["i_max"] + 1e-9
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
        expected_clamped = max(
            -effort_limit, min(record["pid_raw_output"], effort_limit)
        )
        if not math.isclose(
            record["effort_clamped_output"],
            expected_clamped,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise PidDiagnosticContractError(f"effort clamp identity failed for {name}")
        expected_effort_clamp = not math.isclose(
            record["pid_raw_output"], expected_clamped, rel_tol=1e-12, abs_tol=1e-12
        )
        if record["effort_clamp_active"] != expected_effort_clamp:
            raise PidDiagnosticContractError(f"effort clamp flag mismatch for {name}")
        if not math.isclose(
            record["effort_clamped_output"],
            qfrc,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise PidDiagnosticContractError(
                f"effort_clamped_output != qfrc_applied for {name}"
            )

    return {name: decoded_by_name[name] for name in WHEEL_DOF_ORDER}


def runtime_velocity_pid_metadata(
    diagnostics: dict[str, dict[str, Any]], source_topic: str
) -> dict[str, Any]:
    per_wheel = {
        name: {
            "kp": record["kp"],
            "ki": record["ki"],
            "kd": record["kd"],
            "i_max": record["i_max"],
            "i_min": record["i_min"],
            "antiwindup": record["antiwindup"],
            "effort_limit_nm": record["effort_limit_nm"],
        }
        for name, record in diagnostics.items()
    }
    return {
        "schema_version": PID_RUNTIME_METADATA_SCHEMA_VERSION,
        "source_topic": source_topic,
        "per_wheel": per_wheel,
    }


def axis_defaults(axis: str) -> dict[str, str]:
    if axis not in AXIS_CONTRACTS:
        raise ValueError(f"unsupported axis: {axis!r}")
    contract = AXIS_CONTRACTS[axis]
    return {
        "model_file": f"mujoco/rbkairos_scene_tendon_test_{axis}.xml",
        "tendon_name": contract["tendon_name"],
        "sensor_name": contract["sensor_name"],
        "attach_site_name": contract["attach_site_name"],
        "anchor_site_name": contract["anchor_site_name"],
    }


def command_profile_at(
    elapsed_s: float,
    warmup_s: float,
    ramp_s: float,
    post_stop_s: float,
    *,
    constant_end_s: Optional[float] = None,
) -> tuple[str, float, bool]:
    """Return phase and command while an adaptive monitor controls HOLD length."""
    if elapsed_s < warmup_s:
        return "warmup", 0.0, False
    ramp_elapsed = elapsed_s - warmup_s
    if ramp_elapsed < ramp_s:
        phase_fraction = max(0.0, min(1.0, ramp_elapsed / ramp_s))
        return "ramp", 0.5 * (1.0 - math.cos(math.pi * phase_fraction)), False
    if constant_end_s is None or elapsed_s < constant_end_s:
        return "constant", 1.0, False
    post_elapsed = elapsed_s - constant_end_s
    if post_elapsed < post_stop_s:
        return "post_stop", 0.0, False
    return "complete", 0.0, True


def stamp_to_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quaternion_to_rpy(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _array_value(values: Sequence[Any], index: int, default: float = math.nan) -> Any:
    return values[index] if index < len(values) else default


def _parameter_value_to_python(value: Any) -> Any:
    if value.type == ParameterType.PARAMETER_NOT_SET:
        return None
    if value.type == ParameterType.PARAMETER_BOOL:
        return bool(value.bool_value)
    if value.type == ParameterType.PARAMETER_INTEGER:
        return int(value.integer_value)
    if value.type == ParameterType.PARAMETER_DOUBLE:
        return float(value.double_value)
    if value.type == ParameterType.PARAMETER_STRING:
        return str(value.string_value)
    if value.type == ParameterType.PARAMETER_BYTE_ARRAY:
        return list(value.byte_array_value)
    if value.type == ParameterType.PARAMETER_BOOL_ARRAY:
        return list(value.bool_array_value)
    if value.type == ParameterType.PARAMETER_INTEGER_ARRAY:
        return list(value.integer_array_value)
    if value.type == ParameterType.PARAMETER_DOUBLE_ARRAY:
        return list(value.double_array_value)
    if value.type == ParameterType.PARAMETER_STRING_ARRAY:
        return list(value.string_array_value)
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            block = file_obj.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _git_snapshot(workdir: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=workdir,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def resolve_package_path(value: str, package_share: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = package_share / path
    return path.resolve()


def read_controller_facts(urdf_path: Path) -> dict[str, Any]:
    root = ET.parse(urdf_path).getroot()
    ros2_control = root.find("./ros2_control")
    if ros2_control is None:
        raise RuntimeError(f"missing ros2_control in {urdf_path}")
    wheel_records: list[dict[str, float]] = []
    for wheel_name in WHEEL_DOF_ORDER:
        control_joint = ros2_control.find(f"./joint[@name='{wheel_name}']")
        physical_joint = root.find(f"./joint[@name='{wheel_name}']")
        if control_joint is None or physical_joint is None:
            raise RuntimeError(f"missing wheel joint {wheel_name} in {urdf_path}")
        parameters = {
            element.get("name", ""): float(element.text or "nan")
            for element in control_joint.findall("./param")
        }
        limit = physical_joint.find("./limit")
        if limit is None:
            raise RuntimeError(f"missing effort limit for {wheel_name}")
        wheel_records.append(
            {
                "velocity_kp": parameters["velocity_kp"],
                "velocity_ki": parameters["velocity_ki"],
                "velocity_i_max": parameters["velocity_i_max"],
                "velocity_i_min": parameters["velocity_i_min"],
                "effort_limit_nm": float(limit.get("effort", "nan")),
            }
        )
    keys = wheel_records[0]
    for record in wheel_records[1:]:
        if record != keys:
            raise RuntimeError("wheel controller parameters are not uniform")
    return {
        "wheel_velocity_kp": keys["velocity_kp"],
        "wheel_velocity_ki": keys["velocity_ki"],
        "wheel_effort_limit_nm": keys["effort_limit_nm"],
        "wheel_integral_limit": max(abs(keys["velocity_i_min"]), abs(keys["velocity_i_max"])),
        "per_wheel": {
            name: record for name, record in zip(WHEEL_DOF_ORDER, wheel_records)
        },
    }


def inspect_mujoco_model(model_path: Path, names: dict[str, str]) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    tendon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, names["tendon_name"])
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, names["sensor_name"])
    attach_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, names["attach_site_name"])
    anchor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, names["anchor_site_name"])
    if min(tendon_id, sensor_id, attach_id, anchor_id) < 0:
        raise RuntimeError("configured tendon, sensor, or site does not exist in compiled model")
    if (
        model.sensor_type[sensor_id] != mujoco.mjtSensor.mjSENS_TENDONLIMITFRC
        or model.sensor_objid[sensor_id] != tendon_id
    ):
        raise RuntimeError("configured sensor is not the matching tendonlimitfrc")

    pair_friction = [list(map(float, model.pair_friction[index])) for index in range(model.npair)]
    pair_solref_friction = [
        list(map(float, model.pair_solreffriction[index]))
        for index in range(model.npair)
    ]
    if not pair_friction:
        raise RuntimeError("compiled model has no wheel-ground contact pairs")
    rolling_values = {round(values[0], 12) for values in pair_friction}
    lateral_values = {round(values[1], 12) for values in pair_friction}
    if len(rolling_values) != 1 or len(lateral_values) != 1:
        raise RuntimeError("wheel-ground contact friction is not uniform")
    if len({tuple(values) for values in pair_solref_friction}) != 1:
        raise RuntimeError("wheel-ground friction constraint reference is not uniform")

    ur5_pose: list[float] = []
    for joint_name in UR5_JOINT_ORDER:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"compiled model is missing {joint_name}")
        ur5_pose.append(float(model.qpos0[model.jnt_qposadr[joint_id]]))

    return {
        "attach_initial_world_m": list(map(float, data.site_xpos[attach_id])),
        "anchor_initial_world_m": list(map(float, data.site_xpos[anchor_id])),
        "tendon_length0_m": float(model.tendon_length0[tendon_id]),
        "tendon_lower_limit_m": float(model.tendon_range[tendon_id, 0]),
        "tendon_upper_limit_m": float(model.tendon_range[tendon_id, 1]),
        "configured_slack_m": float(
            model.tendon_range[tendon_id, 1] - model.tendon_length0[tendon_id]
        ),
        "tendon_stiffness_n_per_m": float(model.tendon_stiffness[tendon_id]),
        "tendon_damping_n_s_per_m": float(model.tendon_damping[tendon_id]),
        "tendon_frictionloss_n": float(model.tendon_frictionloss[tendon_id]),
        "solver": {
            "name": mujoco.mjtSolver(model.opt.solver).name.removeprefix("mjSOL_"),
            "integrator": mujoco.mjtIntegrator(model.opt.integrator).name.removeprefix("mjINT_"),
            "cone": mujoco.mjtCone(model.opt.cone).name.removeprefix("mjCONE_"),
            "iterations": int(model.opt.iterations),
            "tolerance": float(model.opt.tolerance),
            "timestep_s": float(model.opt.timestep),
            "tendon_solref_limit": list(map(float, model.tendon_solref_lim[tendon_id])),
            "tendon_solimp_limit": list(map(float, model.tendon_solimp_lim[tendon_id])),
            "tendon_margin_m": float(model.tendon_margin[tendon_id]),
        },
        "wheel_ground_friction": {
            "rolling": float(pair_friction[0][0]),
            "lateral": float(pair_friction[0][1]),
            "pair_vectors": pair_friction,
            "solref_friction": pair_solref_friction[0],
        },
        "ur5_joint_pose_rad": ur5_pose,
        "mujoco_version": mujoco.__version__,
    }


class TractionTendonExperiment(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("traction_tendon_experiment")
        self.args = args
        self.package_share = Path(get_package_share_directory("rbkairos"))
        self.model_path = resolve_package_path(args.model_file, self.package_share)
        self.urdf_path = resolve_package_path(args.urdf_file, self.package_share)
        self.names = {
            "tendon_name": args.tendon_name,
            "sensor_name": args.sensor_name,
            "attach_site_name": args.attach_site_name,
            "anchor_site_name": args.anchor_site_name,
        }
        self.model_facts = inspect_mujoco_model(self.model_path, self.names)
        self.controller_facts = read_controller_facts(self.urdf_path)
        self.steady_config = SteadyStateConfig(
            window_s=args.steady_window,
            window_step_s=args.steady_window_step,
            min_observation_s=args.min_observation_duration,
            required_stable_windows=args.required_stable_windows,
            max_wait_s=args.max_wait_duration,
            min_tendon_active_ratio=args.min_active_ratio,
            max_force_cv=args.max_force_cv,
            max_abs_force_slope_n_s=args.max_force_slope,
            max_constraint_extension_m=args.max_constraint_extension,
            max_abs_base_velocity_mps=args.max_steady_base_velocity,
            max_abs_i_term_slope_nm_s=args.max_i_term_slope,
            min_i_clamp_ratio=args.min_i_clamp_ratio,
        )

        created = datetime.now().astimezone()
        run_id = (
            f"{created.strftime('%Y%m%d_%H%M%S')}_axis_{args.axis}_"
            f"{args.solver_variant}_r{args.replicate}"
        )
        if args.run_label:
            run_id += "_" + "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in args.run_label
            )
        self.run_id = run_id
        self.output_dir = Path(args.output_dir).expanduser().resolve() / run_id
        self.output_dir.mkdir(parents=True, exist_ok=False)

        self.started = False
        self.finished = False
        self.acquisition_valid = True
        self.exit_valid = True
        self.reject_reason = ""
        self.start_sim_time: Optional[float] = None
        self.start_wall_monotonic = time.monotonic()
        self.last_state_wall: Optional[float] = None
        self.last_odom_wall: Optional[float] = None
        self.last_dof_wall: Optional[float] = None
        self.last_state_sim_time: Optional[float] = None
        self.latest_state: Optional[dict[str, float]] = None
        self.latest_odom: Optional[dict[str, float]] = None
        self.latest_pid_diagnostics: Optional[dict[str, dict[str, Any]]] = None
        self.latest_ur5_state: Optional[dict[str, list[float]]] = None
        self.ur5_state_error = ""
        self.state_schema_error = ""
        self.pid_diagnostic_error = ""
        self.runtime_parameter_error = ""
        self.runtime_parameters: Optional[dict[str, Any]] = None
        self.runtime_parameter_future: Any = None
        self.wheel_names_seen: set[str] = set()
        self.warmup_tensions: list[float] = []
        self.warmup_active: list[float] = []
        self.warmup_checked = False
        self.initial_base_pose: Optional[dict[str, float]] = None
        self.safety_events: list[dict[str, Any]] = []
        self.last_phase = "preflight"
        self.steady_monitor: Optional[AdaptiveSteadyMonitor] = None
        self.steady_result: Optional[dict[str, Any]] = None
        self.constant_end_elapsed: Optional[float] = None

        self.metadata: dict[str, Any] = self._initial_metadata(created)
        self._open_outputs()
        self._write_metadata()

        self.cmd_publisher = self.create_publisher(Twist, args.cmd_topic, 10)
        self.state_subscription = self.create_subscription(
            Float64MultiArray, args.state_topic, self._on_state, qos_profile_sensor_data
        )
        self.wrench_subscription = self.create_subscription(
            WrenchStamped, args.wrench_topic, self._on_wrench, qos_profile_sensor_data
        )
        self.odom_subscription = self.create_subscription(
            Odometry, args.odom_topic, self._on_odom, qos_profile_sensor_data
        )
        self.dof_subscription = self.create_subscription(
            MujocoDofState, args.dof_topic, self._on_dof, qos_profile_sensor_data
        )
        self.contact_subscription = self.create_subscription(
            MujocoContactState, args.contact_topic, self._on_contact, qos_profile_sensor_data
        )
        parameter_service = "/" + args.mujoco_node.strip("/") + "/get_parameters"
        self.parameter_client = self.create_client(GetParameters, parameter_service)
        self.timer = self.create_timer(1.0 / args.publish_rate, self._on_timer)
        self.get_logger().info(
            f"Tendon recorder preflight: axis={args.axis}, model={self.model_path}, "
            f"output={self.output_dir}"
        )

    def _initial_metadata(self, created: datetime) -> dict[str, Any]:
        profile = {
            "warmup_s": self.args.warmup_duration,
            "ramp_s": self.args.ramp_duration,
            "constant_s": None,
            "min_observation_s": self.args.min_observation_duration,
            "max_wait_s": self.args.max_wait_duration,
            "post_stop_s": self.args.post_stop_duration,
            "constant_start_s": self.args.warmup_duration + self.args.ramp_duration,
            "constant_end_s": None,
            "target_axis_velocity_mps": self.args.target_axis_velocity,
            "ramp_shape": "cosine",
        }
        workspace = Path(__file__).resolve().parents[3]
        metadata = {
            "run_id": self.run_id,
            "created_at": created.isoformat(timespec="seconds"),
            "experiment_type": EXPERIMENT_TYPE,
            "measurement_method": MEASUREMENT_METHOD,
            "schema_version": STATE_SCHEMA_VERSION,
            "state_schema": "traction_tendon_state_v1",
            "summary_schema_version": SUMMARY_SCHEMA_VERSION,
            "axis": self.args.axis,
            "replicate": self.args.replicate,
            "solver_variant": self.args.solver_variant,
            "model_file": str(self.model_path),
            "model_sha256": _sha256(self.model_path),
            "urdf_file": str(self.urdf_path),
            "urdf_sha256": _sha256(self.urdf_path),
            **self.names,
            **{
                key: self.model_facts[key]
                for key in (
                    "attach_initial_world_m",
                    "anchor_initial_world_m",
                    "tendon_length0_m",
                    "tendon_lower_limit_m",
                    "tendon_upper_limit_m",
                    "configured_slack_m",
                    "tendon_stiffness_n_per_m",
                    "tendon_damping_n_s_per_m",
                    "tendon_frictionloss_n",
                    "solver",
                    "wheel_ground_friction",
                    "ur5_joint_pose_rad",
                )
            },
            "controller": self.controller_facts,
            "command_profile": profile,
            "steady_state_contract": {
                "method": STEADY_METHOD,
                **self.steady_config.to_dict(),
            },
            "steady_state_result": None,
            "force_sign_convention": FORCE_SIGN_CONVENTION,
            "topics": {
                "command": self.args.cmd_topic,
                "odom": self.args.odom_topic,
                "state": self.args.state_topic,
                "wrench": self.args.wrench_topic,
                "dof": self.args.dof_topic,
                "contact": self.args.contact_topic,
            },
            "observer_publish_rate_hz_expected": self.args.observer_publish_rate,
            "sampling_contract": {
                "state_expected_rate_hz": self.args.observer_publish_rate,
                "odom_expected_rate_hz": 100.0,
                "min_sample_fraction": 0.8,
                "max_state_gap_s": max(0.05, 3.0 / self.args.observer_publish_rate),
                "max_odom_gap_s": 0.1,
            },
            "runtime_parameters": None,
            "velocity_pid_runtime": None,
            "expected_ur5_preset": self.args.expected_ur5_preset,
            "expected_ur5_pose_rad": (
                list(UR5_PRESETS[self.args.expected_ur5_preset])
                if self.args.expected_ur5_preset else None
            ),
            "ur5_runtime_initial": None,
            "git": _git_snapshot(workspace),
            "runtime_environment": {
                "mujoco_python_version": self.model_facts["mujoco_version"],
            },
            "run_status": {
                "completed": False,
                "valid": True,
                "reject_reason": "",
                "phase": "preflight",
            },
        }
        return metadata

    def _open_outputs(self) -> None:
        self.files: dict[str, Any] = {}
        self.writers: dict[str, csv.DictWriter] = {}

        def open_csv(name: str, fields: Sequence[str]) -> None:
            file_obj = (self.output_dir / name).open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(file_obj, fieldnames=list(fields))
            writer.writeheader()
            self.files[name] = file_obj
            self.writers[name] = writer

        open_csv(
            "cmd_vel.csv",
            (
                "t_experiment",
                "sim_time_s",
                "command_phase",
                "command_scale",
                "axis",
                "target_axis_velocity_mps",
                "cmd_vx",
                "cmd_vy",
                "cmd_wz",
            ),
        )
        open_csv(
            "traction_tendon_state.csv",
            ("t_experiment", "command_phase", *STATE_FIELDS),
        )
        open_csv(
            "traction_tendon_wrench.csv",
            (
                "t_experiment",
                "header_stamp_s",
                "header_frame_id",
                "force_x_n",
                "force_y_n",
                "force_z_n",
                "torque_x_nm",
                "torque_y_nm",
                "torque_z_nm",
            ),
        )
        open_csv(
            "odom.csv",
            (
                "t_experiment",
                "header_stamp_s",
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "vx",
                "vy",
                "vz",
                "wx",
                "wy",
                "wz",
            ),
        )
        wheel_fields = ["t_experiment", "header_stamp_s", "max_abs_qfrc_applied", "nefc", "ncon"]
        for wheel in WHEEL_DOF_ORDER:
            wheel_fields.extend((f"{wheel}_qvel", f"{wheel}_qfrc_applied"))
        open_csv("wheel_qfrc_summary.csv", wheel_fields)
        open_csv(
            "wheel_velocity_pid.csv",
            (
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
            ),
        )
        arm_fields = ["t_experiment", "header_stamp_s"]
        for joint in UR5_JOINT_ORDER:
            arm_fields.extend((f"{joint}_qpos", f"{joint}_qvel"))
        open_csv("ur5_state.csv", arm_fields)
        open_csv(
            "contact_force_summary.csv",
            (
                "t_experiment",
                "header_stamp_s",
                "ncon",
                "normal_force_sum_n",
                "tangent_force_sum_n",
            ),
        )
        open_csv(
            "safety_events.csv",
            ("t_experiment", "sim_time_s", "event", "detail"),
        )

    def _close_outputs(self) -> None:
        for file_obj in self.files.values():
            file_obj.flush()
            file_obj.close()
        self.files.clear()
        self.writers.clear()

    def _write_metadata(self) -> None:
        (self.output_dir / "run_metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _elapsed(self, sim_time: Optional[float] = None) -> float:
        if self.start_sim_time is None:
            return 0.0
        value = sim_time
        if value is None and self.latest_state is not None:
            value = self.latest_state["sim_time_s"]
        if value is None:
            value = self.start_sim_time
        return max(0.0, float(value) - self.start_sim_time)

    def _current_phase(self, sim_time: Optional[float] = None) -> tuple[str, float, bool]:
        return command_profile_at(
            self._elapsed(sim_time),
            self.args.warmup_duration,
            self.args.ramp_duration,
            self.args.post_stop_duration,
            constant_end_s=self.constant_end_elapsed,
        )

    def _record_safety(self, event: str, detail: str) -> None:
        elapsed = self._elapsed()
        sim_time = self.latest_state["sim_time_s"] if self.latest_state else math.nan
        record = {
            "t_experiment": elapsed,
            "sim_time_s": sim_time,
            "event": event,
            "detail": detail,
        }
        self.safety_events.append(record)
        writer = self.writers.get("safety_events.csv")
        if writer is not None:
            writer.writerow(record)

    def _on_state(self, message: Float64MultiArray) -> None:
        now = time.monotonic()
        try:
            if len(message.layout.dim) != 1:
                raise AnalysisInputError("state layout must contain exactly one dimension")
            dimension = message.layout.dim[0]
            if dimension.size != len(STATE_FIELDS) or dimension.stride != len(STATE_FIELDS):
                raise AnalysisInputError("state layout size/stride must both be 17")
            if message.layout.data_offset != 0:
                raise AnalysisInputError("state layout data_offset must be zero")
            state = decode_state_message(dimension.label, message.data)
        except AnalysisInputError as exc:
            self.state_schema_error = str(exc)
            return

        sim_time = state["sim_time_s"]
        if self.last_state_sim_time is not None and sim_time <= self.last_state_sim_time:
            self.state_schema_error = (
                f"state time reversal or duplicate: {sim_time} <= {self.last_state_sim_time}"
            )
            return
        self.last_state_sim_time = sim_time
        self.last_state_wall = now
        self.latest_state = state
        if not self.started:
            return
        elapsed = self._elapsed(sim_time)
        phase, _, _ = self._current_phase(sim_time)
        state_row = {"t_experiment": elapsed, "command_phase": phase, **state}
        self.writers["traction_tendon_state.csv"].writerow(state_row)
        if phase == "constant" and self.steady_monitor is not None:
            self.steady_monitor.add_state(state_row)
        if phase == "warmup":
            self.warmup_tensions.append(state["tension_n"])
            self.warmup_active.append(state["limit_active_flag"])

    def _on_wrench(self, message: WrenchStamped) -> None:
        if not self.started or self.latest_state is None:
            return
        wrench = message.wrench
        header_time = stamp_to_seconds(message.header.stamp)
        self.writers["traction_tendon_wrench.csv"].writerow(
            {
                "t_experiment": self._elapsed(header_time),
                "header_stamp_s": header_time,
                "header_frame_id": message.header.frame_id,
                "force_x_n": wrench.force.x,
                "force_y_n": wrench.force.y,
                "force_z_n": wrench.force.z,
                "torque_x_nm": wrench.torque.x,
                "torque_y_nm": wrench.torque.y,
                "torque_z_nm": wrench.torque.z,
            }
        )

    def _on_odom(self, message: Odometry) -> None:
        self.last_odom_wall = time.monotonic()
        pose = message.pose.pose
        twist = message.twist.twist
        roll, pitch, yaw = quaternion_to_rpy(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        self.latest_odom = {
            "header_stamp_s": stamp_to_seconds(message.header.stamp),
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "vx": float(twist.linear.x),
            "vy": float(twist.linear.y),
            "vz": float(twist.linear.z),
            "wx": float(twist.angular.x),
            "wy": float(twist.angular.y),
            "wz": float(twist.angular.z),
        }
        if self.started:
            header_time = self.latest_odom["header_stamp_s"]
            odom_row = {
                "t_experiment": self._elapsed(header_time), **self.latest_odom
            }
            self.writers["odom.csv"].writerow(odom_row)
            phase, _, _ = self._current_phase(header_time)
            if phase == "constant" and self.steady_monitor is not None:
                self.steady_monitor.add_odom(odom_row)

    def _on_dof(self, message: MujocoDofState) -> None:
        self.last_dof_wall = time.monotonic()
        names = list(message.dof_names)
        qvel = list(message.qvel)
        applied = list(message.qfrc_applied)
        indexed = {name: index for index, name in enumerate(names)}
        self.wheel_names_seen.update(name for name in WHEEL_DOF_ORDER if name in indexed)
        try:
            self.latest_ur5_state = decode_ur5_state(message)
            self.ur5_state_error = ""
        except PidDiagnosticContractError as exc:
            self.ur5_state_error = str(exc)
        try:
            diagnostics = decode_velocity_pid_diagnostics(message)
            current_runtime = runtime_velocity_pid_metadata(diagnostics, self.args.dof_topic)
            frozen_runtime = self.metadata.get("velocity_pid_runtime")
            if self.started and frozen_runtime is not None and current_runtime != frozen_runtime:
                raise PidDiagnosticContractError(
                    "runtime velocity PID configuration changed during the run"
                )
            self.latest_pid_diagnostics = diagnostics
            self.pid_diagnostic_error = ""
        except PidDiagnosticContractError as exc:
            self.pid_diagnostic_error = str(exc)
            return
        if not self.started:
            return
        assert self.latest_pid_diagnostics is not None
        row: dict[str, Any] = {
            "header_stamp_s": stamp_to_seconds(message.header.stamp),
            "nefc": int(message.nefc),
            "ncon": int(message.ncon),
        }
        row["t_experiment"] = self._elapsed(row["header_stamp_s"])
        applied_values: list[float] = []
        for wheel in WHEEL_DOF_ORDER:
            index = indexed.get(wheel, -1)
            velocity = float(_array_value(qvel, index)) if index >= 0 else math.nan
            force = float(_array_value(applied, index)) if index >= 0 else math.nan
            row[f"{wheel}_qvel"] = velocity
            row[f"{wheel}_qfrc_applied"] = force
            if math.isfinite(force):
                applied_values.append(abs(force))
        row["max_abs_qfrc_applied"] = max(applied_values) if applied_values else math.nan
        self.writers["wheel_qfrc_summary.csv"].writerow(row)
        if self.latest_ur5_state is not None:
            arm_row: dict[str, Any] = {
                "t_experiment": row["t_experiment"],
                "header_stamp_s": row["header_stamp_s"],
            }
            for offset, joint in enumerate(UR5_JOINT_ORDER):
                arm_row[f"{joint}_qpos"] = self.latest_ur5_state["positions_rad"][offset]
                arm_row[f"{joint}_qvel"] = self.latest_ur5_state["velocities_rad_s"][offset]
            self.writers["ur5_state.csv"].writerow(arm_row)
        for wheel, diagnostic in self.latest_pid_diagnostics.items():
            pid_row = {
                "t_experiment": row["t_experiment"],
                "header_stamp_s": row["header_stamp_s"],
                "joint_name": wheel,
                **diagnostic,
            }
            self.writers["wheel_velocity_pid.csv"].writerow(pid_row)
            phase, _, _ = self._current_phase(row["header_stamp_s"])
            if phase == "constant" and self.steady_monitor is not None:
                self.steady_monitor.add_pid(pid_row)

    def _on_contact(self, message: MujocoContactState) -> None:
        if not self.started:
            return
        normal = 0.0
        tangent = 0.0
        for index in range(int(message.ncon)):
            start = 6 * index
            force = list(message.force_local_6d[start:start + 6])
            if len(force) < 3:
                continue
            normal += abs(float(force[0]))
            tangent += math.hypot(float(force[1]), float(force[2]))
        self.writers["contact_force_summary.csv"].writerow(
            {
                "t_experiment": self._elapsed(stamp_to_seconds(message.header.stamp)),
                "header_stamp_s": stamp_to_seconds(message.header.stamp),
                "ncon": int(message.ncon),
                "normal_force_sum_n": normal,
                "tangent_force_sum_n": tangent,
            }
        )

    def _poll_runtime_parameters(self) -> None:
        if self.runtime_parameters is not None or self.runtime_parameter_error:
            return
        if self.runtime_parameter_future is None:
            if self.parameter_client.service_is_ready():
                request = GetParameters.Request()
                request.names = list(RUNTIME_PARAMETER_NAMES)
                self.runtime_parameter_future = self.parameter_client.call_async(request)
            return
        if not self.runtime_parameter_future.done():
            return
        try:
            response = self.runtime_parameter_future.result()
            values = [_parameter_value_to_python(value) for value in response.values]
            self.runtime_parameters = dict(zip(RUNTIME_PARAMETER_NAMES, values))
            self._validate_runtime_parameters()
        except Exception as exc:  # noqa: BLE001 - convert ROS/service failures to preflight error
            self.runtime_parameter_error = f"cannot read runtime parameters: {exc}"

    def _validate_runtime_parameters(self) -> None:
        assert self.runtime_parameters is not None
        runtime = self.runtime_parameters
        expected = {
            "traction_tendon.tendon_name": self.args.tendon_name,
            "traction_tendon.sensor_name": self.args.sensor_name,
            "traction_tendon.attach_site_name": self.args.attach_site_name,
            "traction_tendon.anchor_site_name": self.args.anchor_site_name,
            "traction_tendon.state_topic": self.args.state_topic,
            "traction_tendon.wrench_topic": self.args.wrench_topic,
        }
        errors = [
            f"{name}={runtime.get(name)!r}, expected {value!r}"
            for name, value in expected.items()
            if runtime.get(name) != value
        ]
        plugins = runtime.get("physics_plugins")
        if (
            not isinstance(plugins, list)
            or "rbkairos::TractionTendonObserverPlugin" not in plugins
        ):
            errors.append("TractionTendonObserverPlugin is not loaded")
        publish_rate = runtime.get("traction_tendon.publish_rate_hz")
        if not isinstance(publish_rate, (int, float)) or not math.isclose(
            float(publish_rate), self.args.observer_publish_rate, rel_tol=1e-6
        ):
            errors.append(
                f"observer publish rate {publish_rate!r} != {self.args.observer_publish_rate}"
            )
        runtime_model = str(runtime.get("model_file", ""))
        if Path(runtime_model).name != self.model_path.name:
            errors.append(f"runtime model_file {runtime_model!r} != {self.model_path.name!r}")
        if errors:
            self.runtime_parameter_error = "; ".join(errors)
        self.metadata["runtime_parameters"] = runtime
        self._write_metadata()

    def _preflight_errors(self) -> list[str]:
        errors: list[str] = []
        if self.state_schema_error:
            errors.append(self.state_schema_error)
        if self.pid_diagnostic_error:
            errors.append(self.pid_diagnostic_error)
        if self.args.expected_ur5_preset and self.ur5_state_error:
            errors.append(self.ur5_state_error)
        if self.runtime_parameter_error:
            errors.append(self.runtime_parameter_error)
        if self.latest_state is None:
            errors.append(f"no state received on {self.args.state_topic}")
        if self.latest_odom is None:
            errors.append(f"no odometry received on {self.args.odom_topic}")
        if set(WHEEL_DOF_ORDER) - self.wheel_names_seen:
            errors.append("not all four wheel DOFs have been observed")
        if self.latest_pid_diagnostics is None:
            errors.append(f"no velocity PID diagnostics received on {self.args.dof_topic}")
        if self.args.expected_ur5_preset and self.latest_ur5_state is None:
            errors.append(f"no UR5 state received on {self.args.dof_topic}")
        elif self.args.expected_ur5_preset:
            pose_error = validate_ur5_pose(
                self.latest_ur5_state,
                preset=self.args.expected_ur5_preset,
                position_tolerance_rad=self.args.ur5_position_tolerance,
                velocity_tolerance_rad_s=self.args.ur5_velocity_tolerance,
            )
            if pose_error:
                errors.append(pose_error)
        if self.runtime_parameters is None:
            errors.append(f"runtime parameters unavailable from {self.args.mujoco_node}")
        if self.count_subscribers(self.args.cmd_topic) <= 0:
            errors.append(f"no command subscriber on {self.args.cmd_topic}")
        if self.latest_state is not None:
            comparisons = {
                "tendon_length0_m": "tendon_length0_m",
                "lower_limit_m": "tendon_lower_limit_m",
                "upper_limit_m": "tendon_upper_limit_m",
            }
            for state_name, fact_name in comparisons.items():
                if not math.isclose(
                    self.latest_state[state_name],
                    float(self.model_facts[fact_name]),
                    rel_tol=0.0,
                    abs_tol=1e-8,
                ):
                    errors.append(f"state {state_name} does not match compiled model")
        return errors

    def _start(self) -> None:
        assert self.latest_state is not None
        assert self.latest_odom is not None
        assert self.latest_pid_diagnostics is not None
        self.started = True
        self.start_sim_time = self.latest_state["sim_time_s"]
        self.initial_base_pose = dict(self.latest_odom)
        self.metadata["initial_base_pose"] = self.initial_base_pose
        self.metadata["velocity_pid_runtime"] = runtime_velocity_pid_metadata(
            self.latest_pid_diagnostics, self.args.dof_topic
        )
        self.metadata["ur5_runtime_initial"] = self.latest_ur5_state
        self.metadata["run_status"]["phase"] = "warmup"
        self.steady_monitor = AdaptiveSteadyMonitor(
            axis=self.args.axis,
            constant_start_s=(
                self.args.warmup_duration + self.args.ramp_duration
            ),
            config=self.steady_config,
            wheel_names=WHEEL_DOF_ORDER,
        )
        self._write_metadata()
        self.get_logger().info(
            "Preflight passed; starting adaptive Tendon protocol: "
            f"warmup={self.args.warmup_duration:g}s, "
            f"ramp={self.args.ramp_duration:g}s, "
            f"max_wait={self.args.max_wait_duration:g}s"
        )

    def _check_warmup(self) -> Optional[str]:
        if self.warmup_checked:
            return None
        self.warmup_checked = True
        if not self.warmup_tensions:
            return "warmup has no Tendon state samples"
        median_tension = statistics.median(self.warmup_tensions)
        active_ratio = sum(self.warmup_active) / len(self.warmup_active)
        self.metadata["warmup_check"] = {
            "sample_count": len(self.warmup_tensions),
            "median_tension_n": median_tension,
            "active_ratio": active_ratio,
        }
        if median_tension > self.args.max_warmup_tension:
            return f"warmup median tension {median_tension:.6g} N exceeds threshold"
        if active_ratio > self.args.max_warmup_active_ratio:
            return f"warmup active ratio {active_ratio:.6g} exceeds threshold"
        return None

    def _safety_error(self) -> Optional[str]:
        if self.pid_diagnostic_error:
            return self.pid_diagnostic_error
        if self.latest_state is not None:
            if self.latest_state["extension_over_upper_limit_m"] > self.args.abort_extension:
                return "constraint extension exceeds abort threshold"
            if self.latest_state["tension_n"] > self.args.abort_tension:
                return "tendon tension exceeds abort threshold"
        if self.latest_odom is not None and self.initial_base_pose is not None:
            if abs(self.latest_odom["x"] - self.initial_base_pose["x"]) > self.args.max_x:
                return "x displacement exceeds safety bound"
            if abs(self.latest_odom["y"] - self.initial_base_pose["y"]) > self.args.max_y:
                return "y displacement exceeds safety bound"
            if abs(self.latest_odom["yaw"] - self.initial_base_pose["yaw"]) > math.radians(
                self.args.max_yaw_deg
            ):
                return "yaw exceeds safety bound"
        return None

    def _publish_command(self, phase: str, scale: float) -> None:
        command = Twist()
        velocity = self.args.target_axis_velocity * scale
        if self.args.axis == "x":
            command.linear.x = velocity
        else:
            command.linear.y = velocity
        self.cmd_publisher.publish(command)
        if self.started and self.latest_state is not None:
            self.writers["cmd_vel.csv"].writerow(
                {
                    "t_experiment": self._elapsed(),
                    "sim_time_s": self.latest_state["sim_time_s"],
                    "command_phase": phase,
                    "command_scale": scale,
                    "axis": self.args.axis,
                    "target_axis_velocity_mps": self.args.target_axis_velocity,
                    "cmd_vx": command.linear.x,
                    "cmd_vy": command.linear.y,
                    "cmd_wz": command.angular.z,
                }
            )

    def _seal_constant_interval(
        self, elapsed_s: float, steady_result: dict[str, Any]
    ) -> None:
        if self.constant_end_elapsed is not None:
            return
        constant_start = self.args.warmup_duration + self.args.ramp_duration
        self.constant_end_elapsed = max(constant_start, float(elapsed_s))
        profile = self.metadata["command_profile"]
        profile["constant_end_s"] = self.constant_end_elapsed
        profile["constant_s"] = self.constant_end_elapsed - constant_start
        self.steady_result = steady_result
        self.metadata["steady_state_result"] = {
            key: steady_result.get(key)
            for key in (
                "steady_method",
                "status",
                "converged",
                "valid",
                "reject_reasons",
                "stable_start_s",
                "time_to_stable_s",
                "convergence_detected_s",
                "stable_window_count",
                "reset_count",
                "steady_tow_force_n",
                "all_wheels_i_clamped",
                "effort_clamp_any",
                "last_window",
            )
        }
        self._write_metadata()

    def _on_timer(self) -> None:
        if self.finished:
            return
        self._poll_runtime_parameters()
        if not self.started:
            self._publish_command("preflight", 0.0)
            errors = self._preflight_errors()
            if not errors:
                self._start()
                return
            if time.monotonic() - self.start_wall_monotonic > self.args.startup_timeout:
                self._abort("startup_timeout", "; ".join(errors))
            return

        now = time.monotonic()
        if self.last_state_wall is None or now - self.last_state_wall > self.args.topic_timeout:
            self._abort("state_topic_timeout", "Tendon state stopped updating")
            return
        if self.last_odom_wall is None or now - self.last_odom_wall > self.args.topic_timeout:
            self._abort("odom_topic_timeout", "odometry stopped updating")
            return
        if self.last_dof_wall is None or now - self.last_dof_wall > self.args.topic_timeout:
            self._abort("dof_topic_timeout", "MuJoCo DOF state stopped updating")
            return
        safety = self._safety_error()
        if safety:
            self._abort("safety_violation", safety)
            return

        phase, scale, complete = self._current_phase()
        if phase == "ramp" and not self.warmup_checked:
            warmup_error = self._check_warmup()
            if warmup_error:
                self._abort("warmup_preload", warmup_error)
                return
        if phase == "constant":
            assert self.steady_monitor is not None
            decision = self.steady_monitor.advance(self._elapsed())
            if decision["status"] == "effort_clamp":
                self._seal_constant_interval(self._elapsed(), decision)
                self._abort(
                    "effort_clamp",
                    "wheel PID effort clamp became active during constant command",
                )
                return
            if decision["status"] == "not_converged":
                self._seal_constant_interval(self._elapsed(), decision)
                detail = "adaptive steady-state criteria were not sustained before max wait"
                if decision.get("last_window"):
                    detail += f"; last reasons={decision['last_window'].get('reasons', [])}"
                self._abort("not_converged", detail)
                return
            if decision["status"] == "converged":
                self._seal_constant_interval(self._elapsed(), decision)
                phase, scale, complete = self._current_phase()
        self.last_phase = phase
        self.metadata["run_status"]["phase"] = phase
        self._publish_command(phase, scale)
        if complete:
            self._finish(acquisition_valid=True, reject_reason="")

    def _abort(self, event: str, detail: str) -> None:
        if (
            self.started
            and self.constant_end_elapsed is None
            and self.steady_monitor is not None
            and self._elapsed()
            >= self.args.warmup_duration + self.args.ramp_duration
        ):
            self._seal_constant_interval(
                self._elapsed(), self.steady_monitor.result()
            )
        self._record_safety(event, detail)
        self._publish_command("abort_zero", 0.0)
        self._finish(acquisition_valid=False, reject_reason=event)

    def _finish(self, *, acquisition_valid: bool, reject_reason: str) -> None:
        if self.finished:
            return
        self._publish_command("final_zero", 0.0)
        self.finished = True
        self.acquisition_valid = acquisition_valid
        self.metadata["run_status"] = {
            "completed": acquisition_valid,
            "valid": acquisition_valid,
            "reject_reason": reject_reason,
            "phase": "complete" if acquisition_valid else "aborted",
            "safety_event_count": len(self.safety_events),
        }
        self._close_outputs()
        self._write_metadata()

        if acquisition_valid:
            try:
                summary = analyze_traction_tendon_run(
                    self.output_dir,
                    steady_window_s=self.args.steady_window,
                    steady_window_step_s=self.args.steady_window_step,
                    min_observation_s=self.args.min_observation_duration,
                    required_stable_windows=self.args.required_stable_windows,
                    max_wait_s=self.args.max_wait_duration,
                    min_tendon_active_ratio=self.args.min_active_ratio,
                    max_force_cv=self.args.max_force_cv,
                    max_force_slope_n_s=self.args.max_force_slope,
                    max_constraint_extension_m=self.args.max_constraint_extension,
                    max_steady_base_velocity_mps=self.args.max_steady_base_velocity,
                    max_i_term_slope_nm_s=self.args.max_i_term_slope,
                    min_i_clamp_ratio=self.args.min_i_clamp_ratio,
                )
                self.exit_valid = bool(summary["valid"])
                self.metadata["run_status"]["analysis_valid"] = self.exit_valid
                self.metadata["run_status"]["analysis_reject_reason"] = summary["reject_reason"]
                self.metadata["recorder_summary_core"] = {
                    field: summary.get(field) for field in CORE_SUMMARY_FIELDS
                }
                self._write_metadata()
            except AnalysisInputError as exc:
                self.exit_valid = False
                self.metadata["run_status"]["analysis_valid"] = False
                self.metadata["run_status"]["analysis_reject_reason"] = str(exc)
                self._write_metadata()
        else:
            self.exit_valid = False
        self.get_logger().info(
            f"Tendon experiment finished: acquisition_valid={acquisition_valid}, "
            f"analysis_valid={self.exit_valid}, output={self.output_dir}"
        )

    def interrupt(self) -> None:
        if not self.finished:
            self._abort("keyboard_interrupt", "recorder interrupted")


def _positive(parser: argparse.ArgumentParser, name: str, value: float) -> None:
    if value <= 0.0:
        parser.error(f"{name} must be positive")


def _nonnegative(parser: argparse.ArgumentParser, name: str, value: float) -> None:
    if value < 0.0:
        parser.error(f"{name} must be nonnegative")


def parse_args(argv: Optional[list[str]] = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=("x", "y"), default="x")
    parser.add_argument("--velocity-x", type=float)
    parser.add_argument("--velocity-y", type=float)
    parser.add_argument("--warmup-duration", type=float, default=1.0)
    parser.add_argument("--ramp-duration", type=float, default=0.5)
    parser.add_argument("--min-observation-duration", type=float, default=2.0)
    parser.add_argument("--max-wait-duration", type=float, default=60.0)
    parser.add_argument("--post-stop-duration", type=float, default=0.5)
    parser.add_argument("--steady-window", type=float, default=2.0)
    parser.add_argument("--steady-window-step", type=float, default=0.25)
    parser.add_argument("--required-stable-windows", type=int, default=4)
    parser.add_argument("--publish-rate", type=float, default=100.0)
    parser.add_argument("--observer-publish-rate", type=float, default=100.0)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--topic-timeout", type=float, default=1.0)
    parser.add_argument("--max-warmup-tension", type=float, default=1.0)
    parser.add_argument("--max-warmup-active-ratio", type=float, default=0.05)
    parser.add_argument("--min-active-ratio", type=float, default=0.95)
    parser.add_argument("--max-force-cv", type=float, default=0.05)
    parser.add_argument("--max-force-slope", type=float, default=0.5)
    parser.add_argument("--max-constraint-extension", type=float, default=0.005)
    parser.add_argument("--max-steady-base-velocity", type=float, default=0.005)
    parser.add_argument("--max-i-term-slope", type=float, default=0.005)
    parser.add_argument("--min-i-clamp-ratio", type=float, default=0.95)
    parser.add_argument("--abort-extension", type=float, default=0.02)
    parser.add_argument("--abort-tension", type=float, default=2000.0)
    parser.add_argument("--max-x", type=float, default=0.25)
    parser.add_argument("--max-y", type=float, default=0.25)
    parser.add_argument("--max-yaw-deg", type=float, default=10.0)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--solver-variant", default="baseline")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--output-dir", default="data/traction_tendon")
    parser.add_argument("--model-file")
    parser.add_argument("--urdf-file", default="urdf/rbkairos.urdf")
    parser.add_argument("--tendon-name")
    parser.add_argument("--sensor-name")
    parser.add_argument("--attach-site-name")
    parser.add_argument("--anchor-site-name")
    parser.add_argument("--cmd-topic", default=CMD_TOPIC_DEFAULT)
    parser.add_argument("--odom-topic", default=ODOM_TOPIC_DEFAULT)
    parser.add_argument("--state-topic", default=STATE_TOPIC_DEFAULT)
    parser.add_argument("--wrench-topic", default=WRENCH_TOPIC_DEFAULT)
    parser.add_argument("--dof-topic", default=DOF_TOPIC_DEFAULT)
    parser.add_argument("--contact-topic", default=CONTACT_TOPIC_DEFAULT)
    parser.add_argument("--mujoco-node", default=MUJOCO_NODE_DEFAULT)
    parser.add_argument(
        "--expected-ur5-preset",
        choices=tuple(UR5_PRESETS),
        help="require this measured UR5 pose before any nonzero base command",
    )
    parser.add_argument("--ur5-position-tolerance", type=float, default=0.01)
    parser.add_argument("--ur5-velocity-tolerance", type=float, default=0.01)
    args, ros_args = parser.parse_known_args(argv)

    defaults = axis_defaults(args.axis)
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.axis == "x":
        if args.velocity_y not in (None, 0.0):
            parser.error("--velocity-y must be zero or omitted for axis x")
        args.target_axis_velocity = args.velocity_x if args.velocity_x is not None else 0.1
    else:
        if args.velocity_x not in (None, 0.0):
            parser.error("--velocity-x must be zero or omitted for axis y")
        args.target_axis_velocity = args.velocity_y if args.velocity_y is not None else 0.1
    if args.target_axis_velocity == 0.0:
        parser.error("target axis velocity must be nonzero")
    for name in (
        "ramp_duration",
        "min_observation_duration",
        "max_wait_duration",
        "steady_window",
        "steady_window_step",
        "publish_rate",
        "observer_publish_rate",
        "startup_timeout",
        "topic_timeout",
        "abort_extension",
        "abort_tension",
        "max_x",
        "max_y",
        "max_yaw_deg",
    ):
        _positive(parser, f"--{name.replace('_', '-')}", float(getattr(args, name)))
    for name in (
        "warmup_duration",
        "post_stop_duration",
        "max_warmup_tension",
        "max_warmup_active_ratio",
        "min_active_ratio",
        "max_force_cv",
        "max_force_slope",
        "max_constraint_extension",
        "max_steady_base_velocity",
        "max_i_term_slope",
        "min_i_clamp_ratio",
        "ur5_position_tolerance",
        "ur5_velocity_tolerance",
    ):
        _nonnegative(parser, f"--{name.replace('_', '-')}", float(getattr(args, name)))
    if args.steady_window > args.min_observation_duration:
        parser.error("--steady-window cannot exceed --min-observation-duration")
    if args.min_observation_duration > args.max_wait_duration:
        parser.error("--min-observation-duration cannot exceed --max-wait-duration")
    earliest = (
        args.min_observation_duration
        + (args.required_stable_windows - 1) * args.steady_window_step
    )
    if earliest > args.max_wait_duration:
        parser.error("adaptive stability cannot be established before --max-wait-duration")
    if args.required_stable_windows <= 0:
        parser.error("--required-stable-windows must be positive")
    if not 0.0 <= args.min_active_ratio <= 1.0:
        parser.error("--min-active-ratio must be in [0, 1]")
    if not 0.0 <= args.min_i_clamp_ratio <= 1.0:
        parser.error("--min-i-clamp-ratio must be in [0, 1]")
    if args.replicate <= 0:
        parser.error("--replicate must be positive")
    return args, ros_args


def main(argv: Optional[list[str]] = None) -> int:
    args, ros_args = parse_args(argv)
    rclpy.init(args=ros_args)
    node = TractionTendonExperiment(args)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.interrupt()
    finally:
        if not node.finished:
            node.interrupt()
        exit_code = 0 if node.exit_valid else 1
        node.destroy_node()
        rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
