#!/usr/bin/env python3
"""
Pure-Python assets, analysis, and reports for the D2 i_max pilot.

This module is deliberately separate from the formal evaluation package.  D2
collects identifiability evidence; it does not change the production cost or
parameter schema.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

from traction_tendon_steady import (
    SteadyStateConfig,
    SteadyStateError,
    analyze_sustained_steady_state,
)


PILOT_SCHEMA_VERSION = 1
HALF_PI = math.pi / 2.0
UP_POSE_RAD = [0.0, -HALF_PI, 0.0, -HALF_PI, 0.0, 0.0]
WHEEL_JOINTS = (
    "left_front_wheel_joint",
    "right_front_wheel_joint",
    "left_back_wheel_joint",
    "right_back_wheel_joint",
)
UR5_JOINTS = (
    "ur5/shoulder_pan_joint",
    "ur5/shoulder_lift_joint",
    "ur5/elbow_joint",
    "ur5/wrist_1_joint",
    "ur5/wrist_2_joint",
    "ur5/wrist_3_joint",
)
WHEEL_CONTACT_PAIRS = {"fric_lf", "fric_rf", "fric_lb", "fric_rb"}
MJCF_ASSET_FILE_TAGS = {"mesh", "texture", "hfield"}


class PilotInputError(ValueError):
    """Raised when a pilot input violates the frozen D2 contract."""


def _token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _candidate_id(kp: float, ki: float, i_max: float) -> str:
    return f"kp{_token(kp)}_ki{_token(ki)}_imax{_token(i_max)}"


def default_pilot_plan() -> dict[str, Any]:
    """Return the pre-data D2 matrix and provisional offline criteria."""
    candidates = [
        {
            "candidate_id": _candidate_id(50.0, 1.0, i_max),
            "kp": 50.0,
            "ki": 1.0,
            "kd": 0.0,
            "i_max": i_max,
            "i_min": -i_max,
            "effort_limit_nm": 100.0,
        }
        for i_max in (0.5, 1.5, 3.0)
    ]
    durations = {0.01: 60.0, 0.05: 25.0, 0.10: 15.0}
    runs: list[dict[str, Any]] = []
    for candidate in candidates:
        for axis in ("x", "y"):
            for speed in (0.01, 0.05, 0.10):
                for replicate in (1, 2):
                    run_id = (
                        f"axis_{axis}_v{_token(speed)}_"
                        f"{candidate['candidate_id']}_r{replicate:02d}"
                    )
                    runs.append({
                        "run_id": run_id,
                        "candidate_id": candidate["candidate_id"],
                        "axis": axis,
                        "speed_mps": speed,
                        "replicate": replicate,
                        "constant_duration_s": durations[speed],
                    })
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "experiment": "traction_tendon_i_max_identifiability_pilot",
        "status": "planned_before_runtime",
        "fixed": {
            "mu_low": 0.01,
            "mu_high": 0.7,
            "effort_limit_nm": 100.0,
            "kd": 0.0,
            "ur5_preset": "up",
            "ur5_pose_rad": list(UP_POSE_RAD),
        },
        "candidates": candidates,
        "speeds_mps": [0.01, 0.05, 0.10],
        "axes": ["x", "y"],
        "replicates": 2,
        "runs": runs,
        "duration_rationale": {
            "wheel_radius_m": 0.127,
            "blocked_error_estimate_rad_s": {
                str(speed): speed / 0.127 for speed in (0.01, 0.05, 0.10)
            },
            "description": (
                "Durations were frozen before observing D2 physics. They exceed the ideal "
                "i_max/(Ki*speed/radius) clamp time for i_max=3 and include margin for "
                "nonzero wheel motion. Failure to converge remains a result."
            ),
        },
        "analysis": {
            "uses_fixed_final_two_seconds": False,
            "method": "retrospective sustained plateau over the complete constant interval",
            "window_s": 2.0,
            "window_step_s": 0.25,
            "force_slope_limit_n_s": 0.5,
            "i_term_slope_limit_nm_s": 0.005,
            "min_tendon_active_ratio": 0.95,
            "max_force_cv": 0.05,
            "max_constraint_extension_m": 0.005,
            "max_steady_base_velocity_mps": 0.005,
            "min_sustained_i_clamp_s": 0.5,
            "max_repeat_relative_range": 0.03,
            "minimum_low_to_high_effect_ratio": 0.03,
            "minimum_signal_to_repeat_range": 3.0,
            "notes": (
                "Slope thresholds are provisional offline pilot criteria, frozen before data. "
                "Observed distributions are reported for D3; D2 does not alter production gates."
            ),
        },
    }


def write_pilot_plan(plan: Mapping[str, Any], output_dir: Path | str) -> dict[str, str]:
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "pilot_plan.json"
    csv_path = target / "pilot_matrix.csv"
    json_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = (
        "run_id", "candidate_id", "axis", "speed_mps", "replicate",
        "constant_duration_s",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in plan["runs"]:
            writer.writerow({field: row[field] for field in fields})
    return {"plan_json": str(json_path), "matrix_csv": str(csv_path)}


def _rewrite_relative_mjcf_assets(root: ET.Element, source_dir: Path) -> None:
    compiler = root.find("./compiler")
    mesh_base = source_dir
    if compiler is not None and compiler.get("meshdir"):
        raw = Path(str(compiler.get("meshdir")))
        mesh_base = raw if raw.is_absolute() else source_dir / raw
    for element in root.iter():
        raw_file = element.get("file")
        if element.tag not in MJCF_ASSET_FILE_TAGS or not raw_file or "://" in raw_file:
            continue
        path = Path(raw_file)
        if not path.is_absolute():
            base = mesh_base if element.tag == "mesh" else source_dir
            element.set("file", str((base / path).resolve()))


def _params(joint: ET.Element) -> dict[str, ET.Element]:
    return {str(item.get("name")): item for item in joint.findall("param")}


def read_control_initial_pose(root: ET.Element, joint_names: Sequence[str]) -> list[float]:
    values: list[float] = []
    for name in joint_names:
        joint = root.find(f".//ros2_control/joint[@name='{name}']")
        if joint is None:
            raise PilotInputError(f"missing ros2_control joint {name}")
        interface = joint.find("./state_interface[@name='position']")
        value = None if interface is None else interface.find("./param[@name='initial_value']")
        if value is None or value.text is None:
            raise PilotInputError(f"missing position initial_value for {name}")
        values.append(float(value.text))
    return values


def prepare_candidate_assets(
    candidate: Mapping[str, Any],
    output_dir: Path | str,
    *,
    source_urdf: Path | str,
    source_mjcf: Path | str,
    source_scenes: Mapping[str, Path | str],
) -> dict[str, Any]:
    """Create D2-only assets without importing or mutating formal evaluation assets."""
    required = ("candidate_id", "kp", "ki", "kd", "i_max", "i_min", "effort_limit_nm")
    missing = [key for key in required if key not in candidate]
    if missing:
        raise PilotInputError("candidate missing: " + ", ".join(missing))
    if float(candidate["kd"]) != 0.0:
        raise PilotInputError("D2 requires Kd=0")
    if not math.isclose(float(candidate["i_min"]), -float(candidate["i_max"]), abs_tol=1e-12):
        raise PilotInputError("D2 requires i_min=-i_max")
    if float(candidate["effort_limit_nm"]) != 100.0:
        raise PilotInputError("D2 requires effort_limit=100 Nm")

    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    urdf_source = Path(source_urdf).resolve()
    mjcf_source = Path(source_mjcf).resolve()

    urdf_tree = ET.parse(urdf_source)
    urdf_root = urdf_tree.getroot()
    for wheel in WHEEL_JOINTS:
        joint = urdf_root.find(f".//ros2_control/joint[@name='{wheel}']")
        physical = urdf_root.find(f"./joint[@name='{wheel}']/limit")
        if joint is None or physical is None:
            raise PilotInputError(f"missing wheel contract for {wheel}")
        values = _params(joint)
        replacements = {
            "velocity_kp": float(candidate["kp"]),
            "velocity_ki": float(candidate["ki"]),
            "velocity_kd": 0.0,
            "velocity_i_max": float(candidate["i_max"]),
            "velocity_i_min": -float(candidate["i_max"]),
        }
        for name, value in replacements.items():
            if name not in values:
                raise PilotInputError(f"missing {name} for {wheel}")
            values[name].text = str(value)
        if not math.isclose(float(physical.get("effort", "nan")), 100.0, abs_tol=1e-12):
            raise PilotInputError(f"source effort limit for {wheel} is not 100 Nm")
    for name, position in zip(UR5_JOINTS, UP_POSE_RAD):
        joint = urdf_root.find(f".//ros2_control/joint[@name='{name}']")
        if joint is None:
            raise PilotInputError(f"missing arm joint {name}")
        interface = joint.find("./state_interface[@name='position']")
        if interface is None:
            raise PilotInputError(f"missing arm position state interface for {name}")
        value = interface.find("./param[@name='initial_value']")
        if value is None:
            value = ET.SubElement(interface, "param", {"name": "initial_value"})
        value.text = repr(float(position))
    urdf_path = target / "rbkairos.i_max_pilot.urdf"
    urdf_tree.write(urdf_path, encoding="unicode", xml_declaration=True)

    mjcf_tree = ET.parse(mjcf_source)
    mjcf_root = mjcf_tree.getroot()
    _rewrite_relative_mjcf_assets(mjcf_root, mjcf_source.parent)
    seen_pairs: set[str] = set()
    for pair in mjcf_root.findall(".//contact/pair"):
        name = str(pair.get("name"))
        if name not in WHEEL_CONTACT_PAIRS:
            continue
        seen_pairs.add(name)
        values = str(pair.get("friction", "")).split()
        if len(values) < 2:
            raise PilotInputError(f"invalid friction vector for {name}")
        values[0] = "0.01"
        values[1] = "0.7"
        pair.set("friction", " ".join(values))
    if seen_pairs != WHEEL_CONTACT_PAIRS:
        raise PilotInputError("missing four wheel contact pairs")
    mjcf_path = target / "rbkairos.i_max_pilot.xml"
    mjcf_tree.write(mjcf_path, encoding="unicode", xml_declaration=True)

    scene_paths: dict[str, str] = {}
    for axis in ("x", "y"):
        scene_source = Path(source_scenes[axis]).resolve()
        tree = ET.parse(scene_source)
        include = tree.getroot().find("./include")
        if include is None:
            raise PilotInputError(f"scene {axis} has no base include")
        include.set("file", str(mjcf_path))
        path = target / f"rbkairos_scene_tendon_test_{axis}.i_max_pilot.xml"
        tree.write(path, encoding="unicode", xml_declaration=True)
        scene_paths[axis] = str(path)

    manifest = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "candidate": dict(candidate),
        "fixed": default_pilot_plan()["fixed"],
        "urdf_path": str(urdf_path),
        "mjcf_path": str(mjcf_path),
        "scene_paths": scene_paths,
    }
    manifest_path = target / "asset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PilotInputError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise PilotInputError(f"{label} is non-finite")
    return result


def _flag(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and float(value) in (0.0, 1.0):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "1.0", "true"}:
            return True
        if normalized in {"0", "0.0", "false"}:
            return False
    raise PilotInputError(f"{label} must be a boolean flag")


def _window(
    rows: Sequence[Mapping[str, Any]], start: float, end: float
) -> list[Mapping[str, Any]]:
    tolerance = 1e-9
    return [
        row
        for row in rows
        if start - tolerance
        <= _finite(row["t_experiment"], "time")
        <= end + tolerance
    ]


def analyze_time_series(
    state_rows: Sequence[Mapping[str, Any]],
    odom_rows: Sequence[Mapping[str, Any]],
    pid_rows: Sequence[Mapping[str, Any]],
    *,
    axis: str,
    constant_start_s: float,
    constant_end_s: float,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """D2-compatible wrapper around the production sustained-window helper."""
    defaults = dict(default_pilot_plan()["analysis"])
    if thresholds:
        defaults.update(thresholds)
    try:
        config = SteadyStateConfig(
            window_s=float(defaults["window_s"]),
            window_step_s=float(defaults["window_step_s"]),
            min_observation_s=float(defaults["window_s"]),
            # D2's retrospective rule accepts the complete stable tail.  One
            # terminal window is sufficient here because every later window
            # is also required to be stable by the shared offline helper.
            required_stable_windows=1,
            max_wait_s=float(constant_end_s) - float(constant_start_s),
            min_tendon_active_ratio=float(defaults["min_tendon_active_ratio"]),
            max_force_cv=float(defaults["max_force_cv"]),
            max_abs_force_slope_n_s=float(defaults["force_slope_limit_n_s"]),
            max_constraint_extension_m=float(defaults["max_constraint_extension_m"]),
            max_abs_base_velocity_mps=float(defaults["max_steady_base_velocity_mps"]),
            max_abs_i_term_slope_nm_s=float(defaults["i_term_slope_limit_nm_s"]),
            min_i_clamp_ratio=0.95,
        )
        result = analyze_sustained_steady_state(
            state_rows,
            odom_rows,
            pid_rows,
            axis=axis,
            constant_start_s=float(constant_start_s),
            constant_end_s=float(constant_end_s),
            config=config,
            wheel_names=WHEEL_JOINTS,
        )
    except SteadyStateError as exc:
        raise PilotInputError(str(exc)) from exc

    result["thresholds"] = defaults
    result["plateau_force_n"] = result["steady_tow_force_n"]
    if result["reject_reasons"] == ["effort_clamp"]:
        result["reject_reasons"] = ["effort_clamp_active"]
    for metrics in result["per_wheel"].values():
        absolute = metrics.get("time_to_i_clamp_s")
        metrics["time_to_i_clamp_s"] = (
            float(absolute) - float(constant_start_s)
            if absolute is not None else None
        )
    return result


def aggregate_pilot_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for axis in ("x", "y"):
        for speed in (0.01, 0.05, 0.10):
            selected = [
                row for row in rows
                if row.get("axis") == axis
                and math.isclose(float(row.get("speed_mps", math.nan)), speed, abs_tol=1e-12)
            ]
            grouped: dict[float, list[Mapping[str, Any]]] = {}
            for row in selected:
                grouped.setdefault(float(row["i_max"]), []).append(row)
            candidates: list[dict[str, Any]] = []
            cell_valid = len(grouped) == 3
            for i_max in sorted(grouped):
                group = grouped[i_max]
                valid = [
                    row for row in group
                    if row.get("valid") is True
                    and row.get("converged") is True
                    and row.get("effort_clamp_any") is False
                    and row.get("plateau_force_n") is not None
                ]
                forces = [float(row["plateau_force_n"]) for row in valid]
                repeat_range = (
                    (max(forces) - min(forces)) / statistics.median(forces)
                    if len(forces) >= 2 and statistics.median(forces) != 0.0 else math.inf
                )
                candidate_valid = len(valid) == 2
                cell_valid = cell_valid and candidate_valid
                candidates.append({
                    "i_max": i_max,
                    "valid_replicates": len(valid),
                    "forces_n": forces,
                    "median_force_n": statistics.median(forces) if forces else None,
                    "repeat_relative_range": repeat_range,
                    "repeatable": candidate_valid and repeat_range <= 0.03,
                })
            repeatable = cell_valid and all(item["repeatable"] for item in candidates)
            medians = [item["median_force_n"] for item in candidates]
            monotonic = cell_valid and all(
                float(medians[index + 1]) > float(medians[index])
                for index in range(len(medians) - 1)
            )
            if cell_valid:
                low = float(medians[0])
                high = float(medians[-1])
                effect_ratio = (high - low) / abs(low) if low != 0.0 else math.inf
                absolute_noise = max(
                    (max(item["forces_n"]) - min(item["forces_n"]))
                    for item in candidates
                )
                signal_to_repeat = (
                    (high - low) / absolute_noise if absolute_noise > 0.0 else math.inf
                )
            else:
                effect_ratio = math.nan
                signal_to_repeat = math.nan
            sufficient = (
                cell_valid
                and effect_ratio >= 0.03
                and signal_to_repeat >= 3.0
            )
            cells.append({
                "axis": axis,
                "speed_mps": speed,
                "valid": cell_valid,
                "repeatable": repeatable,
                "monotonic": monotonic,
                "sufficient": sufficient,
                "low_to_high_effect_ratio": effect_ratio,
                "signal_to_repeat_range": signal_to_repeat,
                "candidates": candidates,
            })
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "cells": cells,
        "valid_cell_count": sum(item["valid"] for item in cells),
        "all_cells_repeatable": all(item["repeatable"] for item in cells),
        "all_cells_monotonic": all(item["monotonic"] for item in cells),
        "all_cells_sufficient": all(item["sufficient"] for item in cells),
        "i_max_identifiable": all(
            item["valid"] and item["repeatable"] and item["monotonic"] and item["sufficient"]
            for item in cells
        ),
    }


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotInputError(f"{label} must be an object")
    return value


def _require_close(actual: Any, expected: float, label: str, tolerance: float = 1e-9) -> None:
    value = _finite(actual, label)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance):
        raise PilotInputError(f"{label}={value:g}, expected {expected:g}")


def validate_runtime_contract(
    metadata: Mapping[str, Any],
    run_spec: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    """Reject stale 0.65 data and any runtime drift before D2 analysis."""
    if metadata.get("experiment_type") != "traction_tendon_pull":
        raise PilotInputError("experiment_type is not traction_tendon_pull")
    if metadata.get("measurement_method") != "tendon_limit":
        raise PilotInputError("measurement_method is not tendon_limit")
    if metadata.get("schema_version") != 1:
        raise PilotInputError("Tendon schema_version must be 1")
    if metadata.get("axis") != run_spec.get("axis"):
        raise PilotInputError("axis does not match pilot matrix")
    friction = _required_mapping(metadata.get("wheel_ground_friction"), "wheel_ground_friction")
    _require_close(friction.get("rolling"), 0.01, "mu_low")
    _require_close(friction.get("lateral"), 0.7, "mu_high")
    profile = _required_mapping(metadata.get("command_profile"), "command_profile")
    _require_close(
        profile.get("target_axis_velocity_mps"),
        float(run_spec["speed_mps"]),
        "target_axis_velocity_mps",
    )
    duration = _finite(profile.get("constant_end_s"), "constant_end_s") - _finite(
        profile.get("constant_start_s"), "constant_start_s"
    )
    _require_close(duration, float(run_spec["constant_duration_s"]), "constant_duration_s")

    runtime = _required_mapping(metadata.get("velocity_pid_runtime"), "velocity_pid_runtime")
    per_wheel = _required_mapping(runtime.get("per_wheel"), "velocity_pid_runtime.per_wheel")
    if set(per_wheel) != set(WHEEL_JOINTS):
        raise PilotInputError("runtime PID metadata must contain exactly four wheels")
    expected = {
        "kp": float(candidate["kp"]),
        "ki": float(candidate["ki"]),
        "kd": 0.0,
        "i_max": float(candidate["i_max"]),
        "i_min": -float(candidate["i_max"]),
        "effort_limit_nm": 100.0,
    }
    for wheel in WHEEL_JOINTS:
        record = _required_mapping(per_wheel[wheel], f"PID {wheel}")
        for field, value in expected.items():
            _require_close(record.get(field), value, f"{wheel}.{field}")

    if metadata.get("expected_ur5_preset") != "up":
        raise PilotInputError("expected_ur5_preset must be up")
    initial = _required_mapping(metadata.get("ur5_runtime_initial"), "ur5_runtime_initial")
    positions = list(initial.get("positions_rad", []))
    velocities = list(initial.get("velocities_rad_s", []))
    if len(positions) != 6 or len(velocities) != 6:
        raise PilotInputError("UR5 runtime initial state must contain six joints")
    for index, (position, expected_position) in enumerate(zip(positions, UP_POSE_RAD)):
        _require_close(position, expected_position, f"UR5 up position {index}", tolerance=0.01)
    if max(abs(_finite(value, "UR5 initial speed")) for value in velocities) > 0.01:
        raise PilotInputError("UR5 initial speed exceeds 0.01 rad/s")
    run_status = _required_mapping(metadata.get("run_status"), "run_status")
    if run_status.get("completed") is not True or run_status.get("valid") is not True:
        raise PilotInputError("recorder acquisition is not valid and complete")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotInputError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PilotInputError(f"{path} must contain an object")
    return value


def _read_csv(path: Path, required: Iterable[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as file_obj:
            reader = csv.DictReader(file_obj)
            fields = set(reader.fieldnames or [])
            missing = set(required) - fields
            if missing:
                raise PilotInputError(f"{path.name} missing columns: {sorted(missing)}")
            rows = list(reader)
    except OSError as exc:
        raise PilotInputError(f"cannot read {path}: {exc}") from exc
    if not rows:
        raise PilotInputError(f"{path.name} is empty")
    return rows


def _validate_ur5_trace(
    rows: Sequence[Mapping[str, Any]], start: float, end: float
) -> dict[str, Any]:
    selected = _window(rows, start, end)
    if not selected:
        raise PilotInputError("ur5_state.csv has no constant-interval samples")
    max_position_error = 0.0
    max_velocity = 0.0
    for row in selected:
        for joint, expected in zip(UR5_JOINTS, UP_POSE_RAD):
            max_position_error = max(
                max_position_error,
                abs(_finite(row[f"{joint}_qpos"], f"{joint} qpos") - expected),
            )
            max_velocity = max(
                max_velocity,
                abs(_finite(row[f"{joint}_qvel"], f"{joint} qvel")),
            )
    return {
        "sample_count": len(selected),
        "max_position_error_rad": max_position_error,
        "max_abs_velocity_rad_s": max_velocity,
        "valid": max_position_error <= 0.01 and max_velocity <= 0.01,
    }


def _svg_polyline(
    path: Path,
    title: str,
    series: Sequence[tuple[str, Sequence[tuple[float, float]], str]],
) -> None:
    width, height, margin = 960, 440, 60
    finite_points = [
        (x, y) for _, values, _ in series for x, y in values
        if math.isfinite(x) and math.isfinite(y)
    ]
    if finite_points:
        x_min, x_max = min(x for x, _ in finite_points), max(x for x, _ in finite_points)
        y_min, y_max = min(y for _, y in finite_points), max(y for _, y in finite_points)
    else:
        x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0
    if math.isclose(x_min, x_max):
        x_max += 1.0
    if math.isclose(y_min, y_max):
        padding = max(1.0, abs(y_min) * 0.1)
        y_min -= padding
        y_max += padding

    def project(x: float, y: float) -> tuple[float, float]:
        return (
            margin + (x - x_min) * (width - 2 * margin) / (x_max - x_min),
            height - margin - (y - y_min) * (height - 2 * margin) / (y_max - y_min),
        )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="27" text-anchor="middle" font-size="18">{title}</text>',
        (
            f'<line x1="{margin}" y1="{height-margin}" '
            f'x2="{width-margin}" y2="{height-margin}" stroke="black"/>'
        ),
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>',
    ]
    for index, (label, values, color) in enumerate(series):
        points = " ".join(
            f"{px:.2f},{py:.2f}" for px, py in (project(x, y) for x, y in values)
            if math.isfinite(px) and math.isfinite(py)
        )
        lines.append(
            f'<polyline points="{points}" fill="none" '
            f'stroke="{color}" stroke-width="1.4"/>'
        )
        lines.append(
            f'<text x="{margin + index * 205}" y="{height - 15}" '
            f'fill="{color}" font-size="11">{label}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decimate_points(
    points: Sequence[tuple[float, float]], maximum: int = 3000
) -> list[tuple[float, float]]:
    if len(points) <= maximum:
        return list(points)
    stride = max(1, math.ceil(len(points) / maximum))
    result = list(points[::stride])
    if result[-1] != points[-1]:
        result.append(points[-1])
    return result


def analyze_run_directory(
    run_dir: Path | str,
    run_spec: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    write_outputs: bool = True,
) -> dict[str, Any]:
    path = Path(run_dir).resolve()
    metadata = _read_json(path / "run_metadata.json")
    validate_runtime_contract(metadata, run_spec, candidate)
    state = _read_csv(
        path / "traction_tendon_state.csv",
        ("t_experiment", "tension_n", "limit_active_flag", "extension_over_upper_limit_m"),
    )
    odom = _read_csv(path / "odom.csv", ("t_experiment", "vx", "vy", "x", "y", "yaw"))
    pid = _read_csv(
        path / "wheel_velocity_pid.csv",
        (
            "t_experiment", "joint_name", "i_term", "i_clamp_active",
            "effort_clamp_active", "qfrc_applied",
        ),
    )
    arm_required = ["t_experiment"]
    for joint in UR5_JOINTS:
        arm_required.extend((f"{joint}_qpos", f"{joint}_qvel"))
    arm = _read_csv(path / "ur5_state.csv", arm_required)
    profile = metadata["command_profile"]
    constant_start = float(profile["constant_start_s"])
    constant_end = float(profile["constant_end_s"])
    result = analyze_time_series(
        state,
        odom,
        pid,
        axis=str(run_spec["axis"]),
        constant_start_s=constant_start,
        constant_end_s=constant_end,
    )
    arm_health = _validate_ur5_trace(arm, constant_start, constant_end)
    if not arm_health["valid"]:
        result["valid"] = False
        result["reject_reasons"].append("ur5_up_pose_gate")
    result.update({
        "pilot_schema_version": PILOT_SCHEMA_VERSION,
        "run_dir": str(path),
        "run_id": run_spec["run_id"],
        "candidate_id": run_spec["candidate_id"],
        "speed_mps": float(run_spec["speed_mps"]),
        "replicate": int(run_spec["replicate"]),
        "kp": float(candidate["kp"]),
        "ki": float(candidate["ki"]),
        "i_max": float(candidate["i_max"]),
        "mu_low": 0.01,
        "mu_high": 0.7,
        "effort_limit_nm": 100.0,
        "ur5_pose_health": arm_health,
        "source_files": {
            name: {"path": str(path / name), "sha256": sha256(path / name)}
            for name in (
                "run_metadata.json", "traction_tendon_state.csv", "odom.csv",
                "wheel_velocity_pid.csv", "ur5_state.csv",
            )
        },
    })
    if write_outputs:
        (path / "i_max_pilot_run_summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with (path / "i_max_pilot_windows.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file_obj:
            fields = (
                "start_s", "end_s", "sample_count", "active_ratio", "force_median_n",
                "force_mean_n", "force_std_n", "force_cv", "force_slope_n_s",
                "max_constraint_extension_m", "mean_base_velocity_axis_mps", "stable",
            )
            writer = csv.DictWriter(file_obj, fieldnames=fields)
            writer.writeheader()
            for window in result["windows"]:
                writer.writerow({field: window[field] for field in fields})
        _svg_polyline(
            path / "i_max_pilot_tension.svg",
            f"D2 tension {run_spec['axis'].upper()} {run_spec['speed_mps']} m/s",
            [
                (
                    "tension_n",
                    _decimate_points([
                        (
                            _finite(row["t_experiment"], "time"),
                            _finite(row["tension_n"], "tension"),
                        )
                        for row in state
                    ]),
                    "#c62828",
                )
            ],
        )
        colors = ("#1565c0", "#2e7d32", "#ef6c00", "#6a1b9a")
        _svg_polyline(
            path / "i_max_pilot_i_terms.svg",
            "D2 per-wheel I terms",
            [
                (
                    wheel,
                    _decimate_points([
                        (_finite(row["t_experiment"], "time"), _finite(row["i_term"], "I term"))
                        for row in pid if row["joint_name"] == wheel
                    ]),
                    color,
                )
                for wheel, color in zip(WHEEL_JOINTS, colors)
            ],
        )
    return result


def write_matrix_reports(
    summaries: Sequence[Mapping[str, Any]], output_dir: Path | str
) -> dict[str, Any]:
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_pilot_results(summaries)
    aggregate["run_count"] = len(summaries)
    aggregate["fixed"] = default_pilot_plan()["fixed"]
    aggregate["analysis_contract"] = default_pilot_plan()["analysis"]
    stable_windows = [
        window for summary in summaries for window in summary.get("windows", [])
        if window.get("stable") is True
    ]
    aggregate["run_diagnostics"] = {
        "valid_run_count": sum(summary.get("valid") is True for summary in summaries),
        "converged_run_count": sum(summary.get("converged") is True for summary in summaries),
        "all_wheels_i_clamped_run_count": sum(
            summary.get("all_wheels_i_clamped") is True for summary in summaries
        ),
        "effort_clamped_run_count": sum(
            summary.get("effort_clamp_any") is True for summary in summaries
        ),
        "max_time_to_stable_s": max(
            (float(summary["time_to_stable_s"]) for summary in summaries
             if summary.get("time_to_stable_s") is not None),
            default=None,
        ),
        "max_abs_stable_force_slope_n_s": max(
            (abs(float(window["force_slope_n_s"])) for window in stable_windows),
            default=None,
        ),
        "max_stable_force_cv": max(
            (float(window["force_cv"]) for window in stable_windows),
            default=None,
        ),
        "max_constraint_extension_m": max(
            (float(summary["max_constraint_extension_m"]) for summary in summaries
             if summary.get("max_constraint_extension_m") is not None),
            default=None,
        ),
        "max_ur5_position_error_rad": max(
            (float(summary["ur5_pose_health"]["max_position_error_rad"])
             for summary in summaries if summary.get("ur5_pose_health")),
            default=None,
        ),
        "max_ur5_abs_velocity_rad_s": max(
            (float(summary["ur5_pose_health"]["max_abs_velocity_rad_s"])
             for summary in summaries if summary.get("ur5_pose_health")),
            default=None,
        ),
    }
    json_path = target / "i_max_pilot_report.json"
    json_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = target / "i_max_pilot_run_summary.csv"
    fields = (
        "run_id", "candidate_id", "axis", "speed_mps", "replicate", "kp", "ki",
        "i_max", "plateau_force_n", "time_to_stable_s", "all_wheels_i_clamped",
        "effort_clamp_any", "valid", "reject_reasons",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({
                field: (
                    json.dumps(summary.get(field))
                    if field == "reject_reasons"
                    else summary.get(field)
                )
                for field in fields
            })
    force_series: list[tuple[str, list[tuple[float, float]], str]] = []
    colors = ("#1565c0", "#2e7d32", "#ef6c00", "#6a1b9a", "#00838f", "#ad1457")
    for cell, color in zip(aggregate["cells"], colors):
        points = [
            (float(item["i_max"]), float(item["median_force_n"]))
            for item in cell["candidates"] if item["median_force_n"] is not None
        ]
        force_series.append((f"{cell['axis']} {cell['speed_mps']}m/s", points, color))
    _svg_polyline(
        target / "i_max_force_by_condition.svg",
        "Plateau force versus i_max",
        force_series,
    )
    convergence_series: list[tuple[str, list[tuple[float, float]], str]] = []
    clamp_series: list[tuple[str, list[tuple[float, float]], str]] = []
    for cell, color in zip(aggregate["cells"], colors):
        axis, speed = cell["axis"], float(cell["speed_mps"])
        selected = [
            summary for summary in summaries
            if summary.get("axis") == axis
            and math.isclose(float(summary.get("speed_mps", math.nan)), speed, abs_tol=1e-12)
        ]
        convergence_points: list[tuple[float, float]] = []
        clamp_points: list[tuple[float, float]] = []
        for i_max in sorted({float(summary["i_max"]) for summary in selected}):
            group = [summary for summary in selected if float(summary["i_max"]) == i_max]
            stable_times = [
                float(summary["time_to_stable_s"]) for summary in group
                if summary.get("time_to_stable_s") is not None
            ]
            clamp_times = [
                float(wheel["time_to_i_clamp_s"])
                for summary in group for wheel in summary.get("per_wheel", {}).values()
                if wheel.get("time_to_i_clamp_s") is not None
            ]
            if stable_times:
                convergence_points.append((i_max, statistics.median(stable_times)))
            if clamp_times:
                clamp_points.append((i_max, statistics.median(clamp_times)))
        label = f"{axis} {speed}m/s"
        convergence_series.append((label, convergence_points, color))
        clamp_series.append((label, clamp_points, color))
    convergence_path = target / "i_max_time_to_stable.svg"
    clamp_path = target / "i_max_time_to_i_clamp.svg"
    _svg_polyline(
        convergence_path,
        "Retrospective time to stable versus i_max",
        convergence_series,
    )
    _svg_polyline(clamp_path, "Per-wheel time to I clamp versus i_max", clamp_series)
    conclusion = (
        "PASS: i_max is repeatable, monotonic, and sufficiently separated in all six cells."
        if aggregate["i_max_identifiable"]
        else "STOP: D2 did not prove i_max identifiable in every X/Y-speed cell."
    )
    failed_cells = [
        f"{cell['axis'].upper()}/{cell['speed_mps']:.2f} m/s"
        for cell in aggregate["cells"] if not cell["sufficient"]
    ]
    diagnostics = aggregate["run_diagnostics"]
    lines = [
        "# D2 i_max pilot stage conclusion",
        "",
        conclusion,
        "",
        f"- Runs analyzed: {len(summaries)}",
        f"- Valid cells: {aggregate['valid_cell_count']}/6",
        f"- All repeatable: {aggregate['all_cells_repeatable']}",
        f"- All monotonic: {aggregate['all_cells_monotonic']}",
        f"- All sufficiently separated: {aggregate['all_cells_sufficient']}",
        f"- Failed practical-effect cells: {', '.join(failed_cells) if failed_cells else 'none'}",
        (
            "- Valid/converged/all-I-clamped runs: "
            f"{diagnostics['valid_run_count']}/"
            f"{diagnostics['converged_run_count']}/"
            f"{diagnostics['all_wheels_i_clamped_run_count']} of {len(summaries)}"
        ),
        f"- Effort-clamped runs: {diagnostics['effort_clamped_run_count']}",
        f"- Maximum observed time to stable: {diagnostics['max_time_to_stable_s']} s",
        "- Fixed conditions: mu=(0.01,0.7), effort_limit=100 Nm, Kd=0, UR5=up.",
        "- No fixed four-second/final-two-second value was used as the D2 plateau metric.",
        "",
        "D2 stops here. This report does not authorize D3, D4, or formal search.",
    ]
    md_path = target / "D2_stage_conclusion.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "aggregate": aggregate,
        "report_json": str(json_path),
        "summary_csv": str(csv_path),
        "force_plot": str(target / "i_max_force_by_condition.svg"),
        "convergence_plot": str(convergence_path),
        "i_clamp_plot": str(clamp_path),
        "conclusion_md": str(md_path),
    }


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
