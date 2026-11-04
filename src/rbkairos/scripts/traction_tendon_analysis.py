#!/usr/bin/env python3
"""Shared data contract and analysis for MuJoCo traction-tendon runs."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from traction_tendon_steady import (
    STEADY_METHOD,
    SteadyStateConfig,
    SteadyStateError,
    analyze_sustained_steady_state,
)


STATE_SCHEMA_VERSION = 1
SUMMARY_SCHEMA_VERSION = 2
EXPERIMENT_TYPE = "traction_tendon_pull"
MEASUREMENT_METHOD = "tendon_limit"
STATE_FIELDS = (
    "sim_time_s",
    "tendon_length_m",
    "tendon_length0_m",
    "lower_limit_m",
    "upper_limit_m",
    "extension_over_upper_limit_m",
    "tension_n",
    "limit_active_flag",
    "force_x_n",
    "force_y_n",
    "force_z_n",
    "attach_x_m",
    "attach_y_m",
    "attach_z_m",
    "anchor_x_m",
    "anchor_y_m",
    "anchor_z_m",
)
STATE_LAYOUT_LABEL = "traction_tendon_state_v1:" + ",".join(STATE_FIELDS)
AXIS_CONTRACTS = {
    "x": {
        "tendon_name": "traction_cable_x",
        "sensor_name": "traction_cable_x_limit_force",
        "attach_site_name": "traction_attach_x",
        "anchor_site_name": "traction_anchor_x",
        "scene_name": "rbkairos_scene_tendon_test_x.xml",
    },
    "y": {
        "tendon_name": "traction_cable_y",
        "sensor_name": "traction_cable_y_limit_force",
        "attach_site_name": "traction_attach_y",
        "anchor_site_name": "traction_anchor_y",
        "scene_name": "rbkairos_scene_tendon_test_y.xml",
    },
}
STATE_CSV_REQUIRED_COLUMNS = {"t_experiment", *STATE_FIELDS}
STATE_CSV_COLUMNS = ("t_experiment", "command_phase", *STATE_FIELDS)
ODOM_REQUIRED_COLUMNS = {
    "t_experiment",
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "vx",
    "vy",
}
PID_CSV_REQUIRED_COLUMNS = {
    "t_experiment",
    "joint_name",
    "i_term",
    "i_clamp_active",
    "effort_clamp_active",
    "qfrc_applied",
}
RUN_PLOT_FILENAMES = (
    "tension.svg",
    "constraint_extension.svg",
    "base_pose.svg",
    "force_components.svg",
    "wheel_qfrc.svg",
)
SUMMARY_FIELDS = (
    "run_dir",
    "axis",
    "measurement_method",
    "steady_method",
    "steady_converged",
    "time_to_stable_s",
    "stable_window_count",
    "steady_tow_force_n",
    "peak_tow_force_n",
    "steady_tow_force_mean_n",
    "steady_tow_force_std_n",
    "steady_tow_force_cv",
    "steady_force_slope_n_s",
    "steady_i_term_slope_max_nm_s",
    "steady_i_clamp_ratio_min",
    "effort_clamp_any",
    "steady_axis_force_n",
    "steady_cross_axis_force_n",
    "tendon_active_ratio",
    "time_to_tendon_active_s",
    "max_constraint_extension_m",
    "steady_base_velocity_axis_mps",
    "max_base_displacement_axis_m",
    "max_cross_axis_displacement_m",
    "max_yaw_deg_observed",
    "max_pitch_deg_observed",
    "max_roll_deg_observed",
    "wheel_qfrc_applied_max",
    "wheel_saturation_ratio",
    "limit_mode",
    "valid",
    "reject_reason",
    "warnings",
)
CORE_SUMMARY_FIELDS = (
    "steady_tow_force_n",
    "peak_tow_force_n",
    "steady_tow_force_mean_n",
    "steady_tow_force_std_n",
    "steady_tow_force_cv",
    "tendon_active_ratio",
    "max_constraint_extension_m",
    "steady_base_velocity_axis_mps",
    "wheel_qfrc_applied_max",
    "wheel_saturation_ratio",
    "limit_mode",
    "valid",
    "reject_reason",
)
FORCE_SIGN_CONVENTION = (
    "tension_n_is_nonnegative; force_xyz_n_is_force_on_attach_toward_anchor_in_world"
)
WHEEL_DOF_ORDER = (
    "left_front_wheel_joint",
    "right_front_wheel_joint",
    "left_back_wheel_joint",
    "right_back_wheel_joint",
)
PID_DIAGNOSTIC_SCHEMA_VERSION = 1


class AnalysisInputError(RuntimeError):
    """Raised when a run violates the canonical Tendon data contract."""


def _runtime_velocity_pid_config(
    metadata: dict[str, Any],
) -> Optional[dict[str, dict[str, Any]]]:
    runtime = metadata.get("velocity_pid_runtime")
    if runtime is None:
        return None
    if not isinstance(runtime, dict):
        raise AnalysisInputError("velocity_pid_runtime must be an object")
    if runtime.get("schema_version") != PID_DIAGNOSTIC_SCHEMA_VERSION:
        raise AnalysisInputError(
            f"velocity_pid_runtime.schema_version must be {PID_DIAGNOSTIC_SCHEMA_VERSION}"
        )
    if not str(runtime.get("source_topic", "")).strip():
        raise AnalysisInputError("velocity_pid_runtime.source_topic is required")
    per_wheel = runtime.get("per_wheel")
    if not isinstance(per_wheel, dict) or set(per_wheel) != set(WHEEL_DOF_ORDER):
        raise AnalysisInputError(
            "velocity_pid_runtime.per_wheel must contain exactly the four wheel joints"
        )

    fields = ("kp", "ki", "kd", "i_max", "i_min", "effort_limit_nm")
    normalized: dict[str, dict[str, Any]] = {}
    for wheel in WHEEL_DOF_ORDER:
        record = per_wheel.get(wheel)
        if not isinstance(record, dict):
            raise AnalysisInputError(f"velocity_pid_runtime.per_wheel.{wheel} must be an object")
        normalized[wheel] = {
            field: _finite_float(
                record.get(field), f"velocity_pid_runtime.per_wheel.{wheel}.{field}"
            )
            for field in fields
        }
        if not isinstance(record.get("antiwindup"), bool):
            raise AnalysisInputError(
                f"velocity_pid_runtime.per_wheel.{wheel}.antiwindup must be boolean"
            )
        normalized[wheel]["antiwindup"] = bool(record["antiwindup"])
        if normalized[wheel]["i_min"] > normalized[wheel]["i_max"]:
            raise AnalysisInputError(f"velocity PID integral limits are invalid for {wheel}")
        if normalized[wheel]["effort_limit_nm"] <= 0.0:
            raise AnalysisInputError(f"velocity PID effort limit must be positive for {wheel}")

    reference = normalized[WHEEL_DOF_ORDER[0]]
    if any(normalized[wheel] != reference for wheel in WHEEL_DOF_ORDER[1:]):
        raise AnalysisInputError("runtime velocity PID configuration is not uniform")
    return normalized


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisInputError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise AnalysisInputError(f"non-finite value for {label}")
    return result


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise AnalysisInputError(f"{label} must contain {length} values")
    return [_finite_float(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _required_mapping(metadata: dict[str, Any], name: str) -> dict[str, Any]:
    value = metadata.get(name)
    if not isinstance(value, dict):
        raise AnalysisInputError(f"metadata {name} must be an object")
    return value


def steady_config_from_metadata(metadata: dict[str, Any]) -> SteadyStateConfig:
    contract = _required_mapping(metadata, "steady_state_contract")
    if contract.get("method") != STEADY_METHOD:
        raise AnalysisInputError(
            f"steady_state_contract.method must be {STEADY_METHOD!r}"
        )
    fields = set(SteadyStateConfig.__dataclass_fields__)
    if set(contract) != {"method", *fields}:
        missing = sorted(fields - set(contract))
        extra = sorted(set(contract) - fields - {"method"})
        raise AnalysisInputError(
            f"steady_state_contract fields mismatch: missing={missing}, extra={extra}"
        )
    values = {field: contract[field] for field in fields}
    values["required_stable_windows"] = int(
        _finite_float(values["required_stable_windows"], "required_stable_windows")
    )
    try:
        return SteadyStateConfig(**values)
    except (SteadyStateError, TypeError, ValueError) as exc:
        raise AnalysisInputError(f"invalid steady_state_contract: {exc}") from exc


def decode_state_message(layout_label: str, values: Sequence[Any]) -> dict[str, float]:
    """Decode one observer message and enforce the exact schema-v1 contract."""
    if layout_label != STATE_LAYOUT_LABEL:
        raise AnalysisInputError(
            f"state layout label mismatch: expected {STATE_LAYOUT_LABEL!r}, got {layout_label!r}"
        )
    if len(values) != len(STATE_FIELDS):
        raise AnalysisInputError(
            f"state message must contain exactly {len(STATE_FIELDS)} fields, got {len(values)}"
        )
    decoded = {
        name: _finite_float(value, f"state.{name}")
        for name, value in zip(STATE_FIELDS, values)
    }
    if decoded["tension_n"] < 0.0:
        raise AnalysisInputError("state.tension_n must be nonnegative")
    if decoded["limit_active_flag"] not in (0.0, 1.0):
        raise AnalysisInputError("state.limit_active_flag must be exactly 0 or 1")
    return decoded


def validate_metadata(
    metadata: dict[str, Any],
    requested_axis: Optional[str] = None,
    *,
    geometry_tolerance_m: float = 1e-6,
    slack_tolerance_m: float = 1e-8,
) -> None:
    """Validate the canonical run metadata without accepting legacy aliases."""
    if not isinstance(metadata, dict):
        raise AnalysisInputError("run metadata must be a JSON object")
    if metadata.get("experiment_type") != EXPERIMENT_TYPE:
        raise AnalysisInputError(
            f"experiment_type must be {EXPERIMENT_TYPE!r}, got {metadata.get('experiment_type')!r}"
        )
    if metadata.get("measurement_method") != MEASUREMENT_METHOD:
        raise AnalysisInputError(
            f"measurement_method must be {MEASUREMENT_METHOD!r}, "
            f"got {metadata.get('measurement_method')!r}"
        )
    if metadata.get("schema_version") != STATE_SCHEMA_VERSION:
        raise AnalysisInputError(
            f"schema_version must be {STATE_SCHEMA_VERSION}, "
            f"got {metadata.get('schema_version')!r}"
        )
    if metadata.get("state_schema") != "traction_tendon_state_v1":
        raise AnalysisInputError("state_schema must be 'traction_tendon_state_v1'")
    if metadata.get("summary_schema_version") != SUMMARY_SCHEMA_VERSION:
        raise AnalysisInputError(
            f"summary_schema_version must be {SUMMARY_SCHEMA_VERSION}"
        )

    axis = metadata.get("axis")
    if axis not in AXIS_CONTRACTS:
        raise AnalysisInputError(f"axis must be one of {sorted(AXIS_CONTRACTS)}, got {axis!r}")
    contract = AXIS_CONTRACTS[str(axis)]
    for field in ("tendon_name", "sensor_name", "attach_site_name", "anchor_site_name"):
        if metadata.get(field) != contract[field]:
            raise AnalysisInputError(
                f"{field} mismatch for axis {axis}: expected {contract[field]!r}, "
                f"got {metadata.get(field)!r}"
            )
    if requested_axis is not None and requested_axis != axis:
        raise AnalysisInputError(
            f"requested axis {requested_axis!r} does not match metadata axis {axis!r}"
        )

    model_file = str(metadata.get("model_file", ""))
    model_stem = Path(model_file).stem.lower()
    if "tendon" not in model_stem or f"_{axis}" not in model_stem:
        raise AnalysisInputError(
            f"model_file {model_file!r} is not an axis-{axis} Tendon scene"
        )

    attach = _finite_vector(metadata.get("attach_initial_world_m"), 3, "attach_initial_world_m")
    anchor = _finite_vector(metadata.get("anchor_initial_world_m"), 3, "anchor_initial_world_m")
    cable = [anchor[index] - attach[index] for index in range(3)]
    active_index = 0 if axis == "x" else 1
    cross_index = 1 - active_index
    if cable[active_index] >= 0.0:
        raise AnalysisInputError(
            f"axis-{axis} cable must point from attach toward the negative axis"
        )
    if abs(cable[cross_index]) > geometry_tolerance_m or abs(cable[2]) > geometry_tolerance_m:
        raise AnalysisInputError(
            f"axis-{axis} initial cable has a cross-axis or vertical component"
        )

    length0 = _finite_float(metadata.get("tendon_length0_m"), "tendon_length0_m")
    lower = _finite_float(metadata.get("tendon_lower_limit_m"), "tendon_lower_limit_m")
    upper = _finite_float(metadata.get("tendon_upper_limit_m"), "tendon_upper_limit_m")
    slack = _finite_float(metadata.get("configured_slack_m"), "configured_slack_m")
    if length0 <= 0.0 or lower != 0.0 or upper <= length0 or slack <= 0.0:
        raise AnalysisInputError("invalid tendon length/range/slack contract")
    if not math.isclose(upper - length0, slack, rel_tol=0.0, abs_tol=slack_tolerance_m):
        raise AnalysisInputError(
            f"slack mismatch: upper-length0={upper - length0:.12g}, configured={slack:.12g}"
        )
    for field in (
        "tendon_stiffness_n_per_m",
        "tendon_damping_n_s_per_m",
        "tendon_frictionloss_n",
    ):
        if _finite_float(metadata.get(field), field) != 0.0:
            short_name = field.removeprefix("tendon_").split("_")[0]
            raise AnalysisInputError(f"tendon {short_name} must be zero")

    solver = _required_mapping(metadata, "solver")
    if not str(solver.get("name", "")).strip():
        raise AnalysisInputError("metadata solver.name is required")
    if solver.get("cone") is not None and solver.get("cone") != "ELLIPTIC":
        raise AnalysisInputError("solver.cone must be 'ELLIPTIC' for a hardened Tendon run")
    if int(_finite_float(solver.get("iterations"), "solver.iterations")) <= 0:
        raise AnalysisInputError("solver.iterations must be positive")
    if _finite_float(solver.get("timestep_s"), "solver.timestep_s") <= 0.0:
        raise AnalysisInputError("solver.timestep_s must be positive")
    _finite_vector(solver.get("tendon_solref_limit"), 2, "solver.tendon_solref_limit")
    _finite_vector(solver.get("tendon_solimp_limit"), 5, "solver.tendon_solimp_limit")

    controller = _required_mapping(metadata, "controller")
    for field in (
        "wheel_velocity_kp",
        "wheel_velocity_ki",
        "wheel_effort_limit_nm",
        "wheel_integral_limit",
    ):
        _finite_float(controller.get(field), f"controller.{field}")
    _runtime_velocity_pid_config(metadata)
    friction = _required_mapping(metadata, "wheel_ground_friction")
    _finite_float(friction.get("rolling"), "wheel_ground_friction.rolling")
    _finite_float(friction.get("lateral"), "wheel_ground_friction.lateral")
    if friction.get("solref_friction") is not None:
        solref_friction = _finite_vector(
            friction.get("solref_friction"),
            2,
            "wheel_ground_friction.solref_friction",
        )
        if solref_friction != [-1.0, -1.0]:
            raise AnalysisInputError(
                "wheel_ground_friction.solref_friction must be [-1, -1]"
            )
    _finite_vector(metadata.get("ur5_joint_pose_rad"), 6, "ur5_joint_pose_rad")

    profile = _required_mapping(metadata, "command_profile")
    warmup = _finite_float(profile.get("warmup_s"), "command_profile.warmup_s")
    ramp = _finite_float(profile.get("ramp_s"), "command_profile.ramp_s")
    constant = _finite_float(profile.get("constant_s"), "command_profile.constant_s")
    minimum_constant = _finite_float(
        profile.get("min_observation_s"), "command_profile.min_observation_s"
    )
    maximum_constant = _finite_float(
        profile.get("max_wait_s"), "command_profile.max_wait_s"
    )
    post = _finite_float(profile.get("post_stop_s"), "command_profile.post_stop_s")
    constant_start = _finite_float(
        profile.get("constant_start_s"), "command_profile.constant_start_s"
    )
    constant_end = _finite_float(
        profile.get("constant_end_s"), "command_profile.constant_end_s"
    )
    target_velocity = _finite_float(
        profile.get("target_axis_velocity_mps"),
        "command_profile.target_axis_velocity_mps",
    )
    if (
        warmup < 0.0
        or ramp <= 0.0
        or constant <= 0.0
        or minimum_constant <= 0.0
        or maximum_constant < minimum_constant
        or not minimum_constant <= constant <= maximum_constant + 1e-9
        or post < 0.0
    ):
        raise AnalysisInputError("command profile durations are invalid")
    if not math.isclose(constant_start, warmup + ramp, abs_tol=1e-9):
        raise AnalysisInputError("command_profile.constant_start_s is inconsistent")
    if not math.isclose(constant_end, constant_start + constant, abs_tol=1e-9):
        raise AnalysisInputError("command_profile.constant_end_s is inconsistent")
    if target_velocity == 0.0:
        raise AnalysisInputError("command_profile.target_axis_velocity_mps must be nonzero")
    if profile.get("ramp_shape") != "cosine":
        raise AnalysisInputError("command_profile.ramp_shape must be 'cosine'")
    steady_config = steady_config_from_metadata(metadata)
    if not math.isclose(
        steady_config.min_observation_s, minimum_constant, rel_tol=0.0, abs_tol=1e-9
    ):
        raise AnalysisInputError(
            "command_profile.min_observation_s must equal steady min_observation_s"
        )
    if not math.isclose(
        steady_config.max_wait_s, maximum_constant, rel_tol=0.0, abs_tol=1e-9
    ):
        raise AnalysisInputError(
            "command_profile.max_wait_s must equal steady max_wait_s"
        )
    if metadata.get("force_sign_convention") != FORCE_SIGN_CONVENTION:
        raise AnalysisInputError("force_sign_convention mismatch")
    expected_rate = _finite_float(
        metadata.get("observer_publish_rate_hz_expected"),
        "observer_publish_rate_hz_expected",
    )
    if expected_rate <= 0.0:
        raise AnalysisInputError("observer_publish_rate_hz_expected must be positive")
    sampling = _required_mapping(metadata, "sampling_contract")
    for field in ("state_expected_rate_hz", "odom_expected_rate_hz"):
        if _finite_float(sampling.get(field), f"sampling_contract.{field}") <= 0.0:
            raise AnalysisInputError(f"sampling_contract.{field} must be positive")
    fraction = _finite_float(
        sampling.get("min_sample_fraction"), "sampling_contract.min_sample_fraction"
    )
    if not 0.0 < fraction <= 1.0:
        raise AnalysisInputError("sampling_contract.min_sample_fraction must be in (0, 1]")
    for field in ("max_state_gap_s", "max_odom_gap_s"):
        if _finite_float(sampling.get(field), f"sampling_contract.{field}") <= 0.0:
            raise AnalysisInputError(f"sampling_contract.{field} must be positive")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AnalysisInputError(f"missing required file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisInputError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisInputError(f"JSON file {path} must contain an object")
    return value


def _read_csv(
    path: Path,
    required_columns: set[str],
    *,
    exact_columns: Optional[Sequence[str]] = None,
) -> list[dict[str, str]]:
    if not path.is_file():
        raise AnalysisInputError(f"missing required file: {path}")
    try:
        with path.open("r", newline="", encoding="utf-8") as file_obj:
            reader = csv.DictReader(file_obj)
            fieldnames = set(reader.fieldnames or [])
            missing = sorted(required_columns - fieldnames)
            if missing:
                raise AnalysisInputError(f"{path.name} missing columns: {', '.join(missing)}")
            if exact_columns is not None and list(reader.fieldnames or []) != list(exact_columns):
                raise AnalysisInputError(
                    f"{path.name} schema mismatch: expected columns {list(exact_columns)!r}, "
                    f"got {list(reader.fieldnames or [])!r}"
                )
            rows = list(reader)
    except OSError as exc:
        raise AnalysisInputError(f"cannot read CSV file {path}: {exc}") from exc
    if not rows:
        raise AnalysisInputError(f"{path.name} is empty")
    return rows


def _numeric_rows(
    rows: Sequence[dict[str, str]], required_numeric: Iterable[str], source: str
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    last_time: Optional[float] = None
    for row_index, row in enumerate(rows, start=2):
        item: dict[str, Any] = dict(row)
        for field in required_numeric:
            item[field] = _finite_float(row.get(field), f"{source}:{row_index}:{field}")
        current_time = float(item["t_experiment"])
        if last_time is not None and current_time <= last_time:
            raise AnalysisInputError(
                f"time reversal or duplicate in {source} at row {row_index}: "
                f"{current_time} <= {last_time}"
            )
        last_time = current_time
        converted.append(item)
    return converted


def _numeric_pid_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    last_by_wheel: dict[str, float] = {}
    flag_fields = {"i_clamp_active", "effort_clamp_active"}
    numeric = PID_CSV_REQUIRED_COLUMNS - {"joint_name", *flag_fields}
    for row_index, row in enumerate(rows, start=2):
        item: dict[str, Any] = dict(row)
        wheel = str(row.get("joint_name", ""))
        if wheel not in WHEEL_DOF_ORDER:
            raise AnalysisInputError(
                f"wheel_velocity_pid.csv:{row_index}: unexpected joint_name {wheel!r}"
            )
        item["joint_name"] = wheel
        for field in numeric:
            item[field] = _finite_float(
                row.get(field), f"wheel_velocity_pid.csv:{row_index}:{field}"
            )
        for field in flag_fields:
            raw_flag = str(row.get(field, "")).strip().lower()
            if raw_flag in {"1", "1.0", "true"}:
                item[field] = 1.0
            elif raw_flag in {"0", "0.0", "false"}:
                item[field] = 0.0
            else:
                raise AnalysisInputError(
                    f"wheel_velocity_pid.csv:{row_index}:{field} must be 0 or 1"
                )
        timestamp = float(item["t_experiment"])
        previous = last_by_wheel.get(wheel)
        if previous is not None and timestamp <= previous:
            raise AnalysisInputError(
                f"time reversal or duplicate for {wheel} in wheel_velocity_pid.csv"
            )
        last_by_wheel[wheel] = timestamp
        converted.append(item)
    if set(last_by_wheel) != set(WHEEL_DOF_ORDER):
        raise AnalysisInputError(
            "wheel_velocity_pid.csv must contain exactly the four wheel joints"
        )
    return converted


def _mean_std_cv(values: Sequence[float]) -> tuple[float, float, float]:
    if not values:
        raise AnalysisInputError("empty value set")
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    cv = std / abs(mean) if mean != 0.0 else math.inf
    return mean, std, cv


def _median(values: Sequence[float]) -> float:
    if not values:
        raise AnalysisInputError("empty value set")
    return float(statistics.median(values))


def _validate_sample_health(
    rows: Sequence[dict[str, Any]],
    *,
    window_start: float,
    window_end: float,
    expected_rate_hz: float,
    min_sample_fraction: float,
    max_gap_s: float,
    label: str,
) -> dict[str, float | int]:
    if not rows:
        raise AnalysisInputError(f"empty steady window in {label}")
    times = [float(row["t_experiment"]) for row in rows]
    expected_count = expected_rate_hz * (window_end - window_start)
    minimum_count = max(2, math.floor(expected_count * min_sample_fraction))
    if len(times) < minimum_count:
        raise AnalysisInputError(
            f"{label} steady sample count {len(times)} is below required {minimum_count}"
        )
    gaps = [right - left for left, right in zip(times, times[1:])]
    observed_max_gap = max(gaps, default=0.0)
    if observed_max_gap > max_gap_s + 1e-12:
        raise AnalysisInputError(
            f"{label} sample gap {observed_max_gap:.9g} exceeds {max_gap_s:.9g}"
        )
    if times[0] > window_start + max_gap_s or times[-1] < window_end - max_gap_s:
        raise AnalysisInputError(f"{label} does not cover both steady-window boundaries")
    return {
        "sample_count": len(times),
        "expected_sample_count": expected_count,
        "max_gap_s": observed_max_gap,
    }


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    return value


def _write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    (run_dir / "traction_tendon_summary.json").write_text(
        json.dumps(_sanitize_json(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (run_dir / "traction_tendon_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(SUMMARY_FIELDS))
        writer.writeheader()
        writer.writerow(
            {
                field: (
                    json.dumps(summary.get(field), sort_keys=True)
                    if isinstance(summary.get(field), (list, dict))
                    else summary.get(field, "")
                )
                for field in SUMMARY_FIELDS
            }
        )


def _svg_polyline(
    path: Path,
    title: str,
    series: Sequence[tuple[str, Sequence[tuple[float, float]], str]],
) -> None:
    width, height = 900, 420
    margin = 55
    points = [point for _, values, _ in series for point in values]
    finite = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
    if finite:
        x_min, x_max = min(x for x, _ in finite), max(x for x, _ in finite)
        y_min, y_max = min(y for _, y in finite), max(y for _, y in finite)
    else:
        x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    if math.isclose(y_min, y_max):
        padding = max(1.0, abs(y_min) * 0.1)
        y_min -= padding
        y_max += padding

    def project(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        px = margin + (x - x_min) * (width - 2 * margin) / (x_max - x_min)
        py = height - margin - (y - y_min) * (height - 2 * margin) / (y_max - y_min)
        return px, py

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="26" text-anchor="middle" font-size="18">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" '
        f'x2="{width-margin}" y2="{height-margin}" stroke="black"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" '
        f'y2="{height-margin}" stroke="black"/>',
    ]
    for index, (label, values, color) in enumerate(series):
        rendered = " ".join(
            f"{x:.2f},{y:.2f}"
            for x, y in map(project, values)
            if math.isfinite(x) and math.isfinite(y)
        )
        lines.append(
            f'<polyline points="{rendered}" fill="none" '
            f'stroke="{color}" stroke-width="1.5"/>'
        )
        lines.append(
            f'<text x="{margin + index * 160}" y="{height - 14}" '
            f'fill="{color}" font-size="12">{label}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_run_plots(
    run_dir: Path,
    state_rows: Sequence[dict[str, Any]],
    odom_rows: Sequence[dict[str, Any]],
    wheel_rows: Sequence[dict[str, Any]],
) -> None:
    def state_points(field: str) -> list[tuple[float, float]]:
        return [(float(row["t_experiment"]), float(row[field])) for row in state_rows]

    def odom_points(field: str) -> list[tuple[float, float]]:
        return [(float(row["t_experiment"]), float(row[field])) for row in odom_rows]
    _svg_polyline(
        run_dir / "tension.svg",
        "Traction tendon tension",
        [("tension_n", state_points("tension_n"), "#cc2020")],
    )
    _svg_polyline(
        run_dir / "constraint_extension.svg",
        "Constraint extension over upper limit",
        [("extension_m", state_points("extension_over_upper_limit_m"), "#7040c0")],
    )
    _svg_polyline(
        run_dir / "base_pose.svg",
        "Base pose",
        [
            ("x_m", odom_points("x"), "#2050c0"),
            ("y_m", odom_points("y"), "#20a050"),
            ("yaw_rad", odom_points("yaw"), "#d08020"),
        ],
    )
    _svg_polyline(
        run_dir / "force_components.svg",
        "Tendon force components",
        [
            ("force_x_n", state_points("force_x_n"), "#2050c0"),
            ("force_y_n", state_points("force_y_n"), "#20a050"),
            ("force_z_n", state_points("force_z_n"), "#d08020"),
        ],
    )
    wheel_points = [
        (float(row["t_experiment"]), float(row["max_abs_qfrc_applied"]))
        for row in wheel_rows
    ]
    _svg_polyline(
        run_dir / "wheel_qfrc.svg",
        "Maximum absolute wheel qfrc_applied",
        [("qfrc_applied_nm", wheel_points, "#303030")],
    )


def _require_matching_steady_config(
    metadata: dict[str, Any], expected: SteadyStateConfig
) -> None:
    recorded = steady_config_from_metadata(metadata)
    for field, expected_value in expected.to_dict().items():
        actual = getattr(recorded, field)
        if isinstance(expected_value, int):
            matches = actual == expected_value
        else:
            matches = math.isclose(
                float(actual), float(expected_value), rel_tol=0.0, abs_tol=1e-12
            )
        if not matches:
            raise AnalysisInputError(
                f"steady_state_contract.{field}={actual!r}, expected {expected_value!r}"
            )


def analyze_traction_tendon_run(
    run_dir: Path | str,
    *,
    requested_axis: Optional[str] = None,
    steady_window_s: float = 2.0,
    steady_window_step_s: float = 0.25,
    min_observation_s: float = 2.0,
    required_stable_windows: int = 4,
    max_wait_s: float = 60.0,
    min_tendon_active_ratio: float = 0.95,
    max_force_cv: float = 0.05,
    max_force_slope_n_s: float = 0.5,
    max_constraint_extension_m: float = 0.005,
    max_steady_base_velocity_mps: float = 0.005,
    max_i_term_slope_nm_s: float = 0.005,
    min_i_clamp_ratio: float = 0.95,
    min_steady_samples: int = 10,
    write_outputs: bool = True,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Analyze a Tendon run with the production sustained-window contract."""
    try:
        steady_config = SteadyStateConfig(
            window_s=float(steady_window_s),
            window_step_s=float(steady_window_step_s),
            min_observation_s=float(min_observation_s),
            required_stable_windows=int(required_stable_windows),
            max_wait_s=float(max_wait_s),
            min_tendon_active_ratio=float(min_tendon_active_ratio),
            max_force_cv=float(max_force_cv),
            max_abs_force_slope_n_s=float(max_force_slope_n_s),
            max_constraint_extension_m=float(max_constraint_extension_m),
            max_abs_base_velocity_mps=float(max_steady_base_velocity_mps),
            max_abs_i_term_slope_nm_s=float(max_i_term_slope_nm_s),
            min_i_clamp_ratio=float(min_i_clamp_ratio),
        )
    except (SteadyStateError, TypeError, ValueError) as exc:
        raise AnalysisInputError(f"invalid adaptive steady configuration: {exc}") from exc

    run_path = Path(run_dir).resolve()
    metadata = _read_json(run_path / "run_metadata.json")
    validate_metadata(metadata, requested_axis=requested_axis)
    _require_matching_steady_config(metadata, steady_config)

    raw_state = _read_csv(
        run_path / "traction_tendon_state.csv",
        STATE_CSV_REQUIRED_COLUMNS,
        exact_columns=STATE_CSV_COLUMNS,
    )
    state_rows = _numeric_rows(
        raw_state, STATE_CSV_REQUIRED_COLUMNS, "traction_tendon_state.csv"
    )
    for row_index, row in enumerate(state_rows, start=2):
        if row["tension_n"] < 0.0:
            raise AnalysisInputError(f"tension_n must be nonnegative at state row {row_index}")
        if row["limit_active_flag"] not in (0.0, 1.0):
            raise AnalysisInputError(f"limit_active_flag must be 0 or 1 at state row {row_index}")
        for row_field, metadata_field in (
            ("tendon_length0_m", "tendon_length0_m"),
            ("lower_limit_m", "tendon_lower_limit_m"),
            ("upper_limit_m", "tendon_upper_limit_m"),
        ):
            if not math.isclose(
                float(row[row_field]),
                float(metadata[metadata_field]),
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise AnalysisInputError(
                    f"state {row_field} does not match metadata at row {row_index}"
                )

    raw_odom = _read_csv(run_path / "odom.csv", ODOM_REQUIRED_COLUMNS)
    odom_rows = _numeric_rows(raw_odom, ODOM_REQUIRED_COLUMNS, "odom.csv")
    raw_pid = _read_csv(
        run_path / "wheel_velocity_pid.csv", PID_CSV_REQUIRED_COLUMNS
    )
    pid_rows = _numeric_pid_rows(raw_pid)

    wheel_path = run_path / "wheel_qfrc_summary.csv"
    wheel_rows: list[dict[str, Any]] = []
    if wheel_path.is_file():
        raw_wheel = _read_csv(
            wheel_path, {"t_experiment", "max_abs_qfrc_applied"}
        )
        wheel_rows = _numeric_rows(
            raw_wheel,
            {"t_experiment", "max_abs_qfrc_applied"},
            "wheel_qfrc_summary.csv",
        )

    axis = str(metadata["axis"])
    profile = metadata["command_profile"]
    constant_start = float(profile["constant_start_s"])
    constant_end = float(profile["constant_end_s"])
    try:
        adaptive = analyze_sustained_steady_state(
            state_rows,
            odom_rows,
            pid_rows,
            axis=axis,
            constant_start_s=constant_start,
            constant_end_s=constant_end,
            config=steady_config,
            wheel_names=WHEEL_DOF_ORDER,
        )
    except SteadyStateError as exc:
        raise AnalysisInputError(str(exc)) from exc

    tolerance = 1e-9
    stable_start = adaptive.get("stable_start_s")
    plateau_state = (
        [
            row for row in state_rows
            if float(stable_start) - tolerance
            <= row["t_experiment"]
            <= constant_end + tolerance
        ]
        if stable_start is not None else []
    )
    plateau_odom = (
        [
            row for row in odom_rows
            if float(stable_start) - tolerance
            <= row["t_experiment"]
            <= constant_end + tolerance
        ]
        if stable_start is not None else []
    )
    active_rows = [row for row in plateau_state if row["limit_active_flag"] == 1.0]
    if adaptive["converged"] and (
        len(plateau_state) < min_steady_samples
        or len(plateau_odom) < min_steady_samples
    ):
        raise AnalysisInputError("adaptive steady plateau has insufficient samples")

    sample_health: dict[str, Any] = {}
    if adaptive["converged"]:
        sampling = metadata["sampling_contract"]
        sample_health = {
            "state": _validate_sample_health(
                plateau_state,
                window_start=float(stable_start),
                window_end=constant_end,
                expected_rate_hz=float(sampling["state_expected_rate_hz"]),
                min_sample_fraction=float(sampling["min_sample_fraction"]),
                max_gap_s=float(sampling["max_state_gap_s"]),
                label="state",
            ),
            "odom": _validate_sample_health(
                plateau_odom,
                window_start=float(stable_start),
                window_end=constant_end,
                expected_rate_hz=float(sampling["odom_expected_rate_hz"]),
                min_sample_fraction=float(sampling["min_sample_fraction"]),
                max_gap_s=float(sampling["max_odom_gap_s"]),
                label="odom",
            ),
        }

    active_tensions = [float(row["tension_n"]) for row in active_rows]
    if active_tensions:
        force_mean, force_std, force_cv = _mean_std_cv(active_tensions)
        steady_force = _median(active_tensions)
    else:
        force_mean, force_std, force_cv, steady_force = (
            math.nan, math.nan, math.inf, math.nan
        )
    active_ratio = (
        len(active_rows) / len(plateau_state) if plateau_state else 0.0
    )

    command_start = float(profile["warmup_s"])
    command_rows = [
        row for row in state_rows
        if command_start - tolerance <= row["t_experiment"] <= constant_end + tolerance
    ]
    command_odom = [
        row for row in odom_rows
        if command_start - tolerance <= row["t_experiment"] <= constant_end + tolerance
    ]
    if not command_rows or not command_odom:
        raise AnalysisInputError("empty ramp/constant command interval")
    peak_force = max(float(row["tension_n"]) for row in command_rows)
    first_active = next(
        (row for row in command_rows if row["limit_active_flag"] == 1.0), None
    )
    time_to_active = (
        float(first_active["t_experiment"]) - command_start
        if first_active is not None else math.nan
    )
    max_extension = max(
        float(row["extension_over_upper_limit_m"]) for row in command_rows
    )

    target_velocity = float(profile["target_axis_velocity_mps"])
    command_sign = 1.0 if target_velocity > 0.0 else -1.0
    force_axis_field = "force_x_n" if axis == "x" else "force_y_n"
    force_cross_field = "force_y_n" if axis == "x" else "force_x_n"
    steady_axis_force = (
        _median([-command_sign * float(row[force_axis_field]) for row in active_rows])
        if active_rows else math.nan
    )
    steady_cross_force = (
        _median([
            math.hypot(float(row[force_cross_field]), float(row["force_z_n"]))
            for row in active_rows
        ])
        if active_rows else math.nan
    )
    velocity_axis_field = "vx" if axis == "x" else "vy"
    steady_velocity = (
        statistics.fmean(float(row[velocity_axis_field]) for row in plateau_odom)
        if plateau_odom else math.nan
    )

    initial = command_odom[0]
    axis_position = "x" if axis == "x" else "y"
    cross_position = "y" if axis == "x" else "x"
    max_axis_displacement = max(
        abs(float(row[axis_position]) - float(initial[axis_position]))
        for row in command_odom
    )
    max_cross_displacement = max(
        abs(float(row[cross_position]) - float(initial[cross_position]))
        for row in command_odom
    )
    max_yaw = math.degrees(max(
        abs(float(row["yaw"]) - float(initial["yaw"])) for row in command_odom
    ))
    max_pitch = math.degrees(max(
        abs(float(row["pitch"]) - float(initial["pitch"])) for row in command_odom
    ))
    max_roll = math.degrees(max(
        abs(float(row["roll"]) - float(initial["roll"])) for row in command_odom
    ))

    wheel_max = (
        max(float(row["max_abs_qfrc_applied"]) for row in wheel_rows)
        if wheel_rows else math.nan
    )
    runtime_pid = _runtime_velocity_pid_config(metadata)
    wheel_effort_limit = (
        float(runtime_pid[WHEEL_DOF_ORDER[0]]["effort_limit_nm"])
        if runtime_pid is not None
        else float(metadata["controller"]["wheel_effort_limit_nm"])
    )
    wheel_saturation_ratio = (
        wheel_max / wheel_effort_limit
        if math.isfinite(wheel_max) and wheel_effort_limit > 0.0 else math.nan
    )
    if not math.isfinite(wheel_saturation_ratio):
        limit_mode = "unknown"
    elif wheel_saturation_ratio >= 0.98:
        limit_mode = "wheel_effort_limited"
    else:
        limit_mode = "not_effort_limited"

    last_window = adaptive.get("last_window") or {}
    per_wheel_window = last_window.get("per_wheel", {})
    steady_i_slope_max = max(
        (abs(float(item["i_slope_nm_s"])) for item in per_wheel_window.values()),
        default=math.nan,
    )
    steady_i_clamp_min = min(
        (float(item["i_clamp_ratio"]) for item in per_wheel_window.values()),
        default=math.nan,
    )
    reject_reasons = list(adaptive["reject_reasons"])
    if not adaptive["converged"] and last_window.get("reasons"):
        reject_reasons.append(
            "unstable_signals=" + ",".join(last_window["reasons"])
        )
    if max_extension > max_constraint_extension_m:
        reject_reasons.append(
            f"constraint_extension={max_extension:.6g}>{max_constraint_extension_m:.6g}"
        )
    run_status = metadata.get("run_status")
    if isinstance(run_status, dict) and not bool(run_status.get("valid", True)):
        reject_reasons.append(
            f"recorder_invalid={run_status.get('reject_reason', 'unspecified')}"
        )

    summary: dict[str, Any] = {
        "run_dir": str(run_path),
        "axis": axis,
        "measurement_method": MEASUREMENT_METHOD,
        "steady_method": STEADY_METHOD,
        "steady_converged": bool(adaptive["converged"]),
        "time_to_stable_s": adaptive.get("time_to_stable_s"),
        "stable_window_count": adaptive.get("stable_window_count", 0),
        "steady_tow_force_n": steady_force if adaptive["converged"] else math.nan,
        "peak_tow_force_n": peak_force,
        "steady_tow_force_mean_n": force_mean,
        "steady_tow_force_std_n": force_std,
        "steady_tow_force_cv": force_cv,
        "steady_force_slope_n_s": last_window.get("force_slope_n_s", math.nan),
        "steady_i_term_slope_max_nm_s": steady_i_slope_max,
        "steady_i_clamp_ratio_min": steady_i_clamp_min,
        "effort_clamp_any": bool(adaptive["effort_clamp_any"]),
        "steady_axis_force_n": steady_axis_force,
        "steady_cross_axis_force_n": steady_cross_force,
        "tendon_active_ratio": active_ratio,
        "time_to_tendon_active_s": time_to_active,
        "max_constraint_extension_m": max_extension,
        "steady_base_velocity_axis_mps": steady_velocity,
        "max_base_displacement_axis_m": max_axis_displacement,
        "max_cross_axis_displacement_m": max_cross_displacement,
        "max_yaw_deg_observed": max_yaw,
        "max_pitch_deg_observed": max_pitch,
        "max_roll_deg_observed": max_roll,
        "wheel_qfrc_applied_max": wheel_max,
        "wheel_saturation_ratio": wheel_saturation_ratio,
        "limit_mode": limit_mode,
        "valid": not reject_reasons,
        "reject_reason": "; ".join(reject_reasons),
        "warnings": [] if wheel_rows else ["wheel_qfrc_summary.csv unavailable"],
        "analysis": {
            "steady_method": STEADY_METHOD,
            "stable_start_s": stable_start,
            "steady_end_s": constant_end,
            "convergence_detected_s": adaptive.get("convergence_detected_s"),
            "steady_sample_count": len(plateau_state),
            "steady_active_sample_count": len(active_rows),
            "sample_health": sample_health,
            "steady_state_config": steady_config.to_dict(),
            "last_window": last_window,
            "per_wheel": adaptive.get("per_wheel", {}),
        },
    }
    if write_outputs:
        _write_summary(run_path, summary)
    if make_plots:
        _write_run_plots(run_path, state_rows, odom_rows, wheel_rows)
    return summary


def discover_run_dirs(input_dir: Path | str) -> list[Path]:
    """Discover only canonical Tendon runs; legacy Spring directories are ignored."""
    root = Path(input_dir).resolve()
    candidates = [root] if (root / "run_metadata.json").is_file() else []
    candidates.extend(path.parent for path in root.rglob("run_metadata.json"))
    discovered: list[Path] = []
    for candidate in sorted(set(candidates)):
        metadata: dict[str, Any] = {}
        state_exists = (candidate / "traction_tendon_state.csv").is_file()
        try:
            metadata = _read_json(candidate / "run_metadata.json")
            claims_tendon = (
                metadata.get("experiment_type") == EXPERIMENT_TYPE
                or metadata.get("measurement_method") == MEASUREMENT_METHOD
            )
            if not state_exists:
                if claims_tendon:
                    raise AnalysisInputError("missing traction_tendon_state.csv")
                continue
            validate_metadata(metadata)
        except AnalysisInputError as exc:
            if (
                metadata.get("experiment_type") == EXPERIMENT_TYPE
                or metadata.get("measurement_method") == MEASUREMENT_METHOD
            ):
                raise AnalysisInputError(
                    f"malformed Tendon run {candidate}: {exc}"
                ) from exc
            continue
        discovered.append(candidate)
    return sorted(discovered)


def _write_batch_outputs(output_dir: Path, summaries: Sequence[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "traction_tendon_batch_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(SUMMARY_FIELDS))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    field: (
                        json.dumps(summary.get(field), sort_keys=True)
                        if isinstance(summary.get(field), (list, dict))
                        else summary.get(field, "")
                    )
                    for field in SUMMARY_FIELDS
                }
            )

    groups: dict[str, dict[str, Any]] = {}
    for axis in AXIS_CONTRACTS:
        baseline_summaries = [
            summary
            for summary in summaries
            if summary.get("axis") == axis
            and _read_json(Path(str(summary["run_dir"])) / "run_metadata.json").get(
                "solver_variant", "baseline"
            )
            == "baseline"
            and math.isfinite(float(summary["steady_tow_force_n"]))
        ]
        values = [float(summary["steady_tow_force_n"]) for summary in baseline_summaries]
        if not values:
            continue
        median = _median(values)
        relative_range = (max(values) - min(values)) / abs(median) if median else math.inf
        groups[axis] = {
            "run_count": len(values),
            "valid_run_count": sum(
                summary.get("valid") is True for summary in baseline_summaries
            ),
            "steady_tow_force_median_n": median,
            "steady_tow_force_min_n": min(values),
            "steady_tow_force_max_n": max(values),
            "relative_range": relative_range,
        }
    sensitivity: dict[str, list[dict[str, Any]]] = {}
    for axis, baseline in groups.items():
        reference = float(baseline["steady_tow_force_median_n"])
        comparisons: list[dict[str, Any]] = []
        for summary in summaries:
            if summary.get("axis") != axis:
                continue
            metadata = _read_json(Path(str(summary["run_dir"])) / "run_metadata.json")
            variant = str(metadata.get("solver_variant", "baseline"))
            if variant == "baseline":
                continue
            value = float(summary["steady_tow_force_n"])
            comparisons.append(
                {
                    "solver_variant": variant,
                    "run_dir": summary["run_dir"],
                    "steady_tow_force_n": value,
                    "relative_change_from_baseline": abs(value - reference) / abs(reference),
                    "run_valid": summary.get("valid") is True,
                }
            )
        if comparisons:
            sensitivity[axis] = comparisons
    report = {
        "measurement_method": MEASUREMENT_METHOD,
        "schema_version": STATE_SCHEMA_VERSION,
        "run_count": len(summaries),
        "groups": groups,
        "solver_sensitivity": sensitivity,
    }
    (output_dir / "traction_tendon_batch_report.json").write_text(
        json.dumps(_sanitize_json(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    series = []
    colors = {"x": "#2050c0", "y": "#20a050"}
    for axis in AXIS_CONTRACTS:
        points = [
            (float(index + 1), float(summary["steady_tow_force_n"]))
            for index, summary in enumerate(summaries)
            if summary.get("axis") == axis
            and _read_json(Path(str(summary["run_dir"])) / "run_metadata.json").get(
                "solver_variant", "baseline"
            )
            == "baseline"
            and math.isfinite(float(summary["steady_tow_force_n"]))
        ]
        if points:
            series.append((f"axis_{axis}", points, colors[axis]))
    _svg_polyline(
        output_dir / "traction_tendon_repeatability.svg",
        "Traction tendon repeatability",
        series,
    )


def analyze_traction_tendon_batch(
    input_dir: Path | str,
    *,
    output_dir: Optional[Path | str] = None,
    **analysis_options: Any,
) -> list[dict[str, Any]]:
    run_dirs = discover_run_dirs(input_dir)
    if not run_dirs:
        raise AnalysisInputError(f"no canonical Tendon runs found under {input_dir}")
    summaries = [
        analyze_traction_tendon_run(run_dir, **analysis_options) for run_dir in run_dirs
    ]
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else Path(input_dir).resolve()
    )
    _write_batch_outputs(destination, summaries)
    return summaries
