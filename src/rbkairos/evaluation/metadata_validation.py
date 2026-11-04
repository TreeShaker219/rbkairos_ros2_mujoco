from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.parameters import (
    FIXED_EFFORT_LIMIT_NM,
    FIXED_KD,
    FIXED_MU_HIGH,
    FIXED_MU_LOW,
    EvaluationParameters,
)
from evaluation.traction_conditions import (
    TRACTION_CONDITION_IDS,
    condition_axis,
    condition_speed_mps,
)


WHEEL_JOINTS = [
    "left_front_wheel_joint",
    "right_front_wheel_joint",
    "left_back_wheel_joint",
    "right_back_wheel_joint",
]
WHEEL_CONTACT_PAIRS = ["fric_lf", "fric_rf", "fric_lb", "fric_rb"]
WHEEL_GROUND_GEOMS = ["capsule_lf", "capsule_rf", "capsule_lb", "capsule_rb"]
PRIMARY_FRICTION_INDEX = 1
ABS_TOLERANCE = 1e-9
REL_TOLERANCE = 1e-9


@dataclass
class RuntimeMetadataValidation:
    valid: bool
    reject_reasons: list[str]
    checks: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "reject_reasons": list(self.reject_reasons),
            "checks": self.checks,
        }


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing file: {path}"
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"cannot read JSON: {path}: {exc}"
    if not isinstance(data, dict):
        return None, f"JSON root is not an object: {path}"
    return data, None


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _matches(actual: Any, expected: float) -> bool:
    actual_float = _float_or_none(actual)
    if actual_float is None:
        return False
    return math.isclose(
        actual_float,
        float(expected),
        rel_tol=REL_TOLERANCE,
        abs_tol=ABS_TOLERANCE,
    )


def _mismatch(
    parameter: str,
    expected: float,
    actual: Any,
    location: str,
) -> dict[str, Any]:
    return {
        "parameter": parameter,
        "expected": float(expected),
        "actual": actual,
        "location": location,
    }


def _controller_facts(metadata: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    velocity_pid_runtime = metadata.get("velocity_pid_runtime")
    if isinstance(velocity_pid_runtime, dict):
        return velocity_pid_runtime, "velocity_pid_runtime"
    runtime_configuration = metadata.get("runtime_configuration")
    if isinstance(runtime_configuration, dict):
        controller = runtime_configuration.get("controller")
        if isinstance(controller, dict):
            return controller, "runtime_configuration.controller"

    controller = metadata.get("controller_facts")
    if isinstance(controller, dict):
        return controller, "controller_facts"

    return None, None


def _validate_controller_parameters(
    metadata: dict[str, Any],
    metadata_path: Path,
    parameters: EvaluationParameters,
    scope: str,
) -> tuple[dict[str, Any], str | None]:
    check: dict[str, Any] = {
        "kind": "wheel_velocity_pid",
        "scope": scope,
        "metadata_path": str(metadata_path),
        "status": "unknown",
        "source": None,
        "snapshot_status": None,
        "checked": [],
        "missing": [],
        "mismatches": [],
        "notes": [],
    }
    controller, source = _controller_facts(metadata)
    if controller is None:
        check["status"] = "missing"
        check["missing"].append("controller_facts")
        return check, f"runtime_metadata.{scope}.pid_contract_unverifiable"

    check["source"] = source
    snapshot = controller.get("runtime_snapshot")
    if isinstance(snapshot, dict):
        check["snapshot_status"] = snapshot.get("status")
        errors = snapshot.get("errors")
        if errors:
            check["notes"].append({"runtime_snapshot_errors": errors})

    expected_values = {
        "kp": float(parameters.kp),
        "ki": float(parameters.ki),
        "kd": FIXED_KD,
        "i_max": float(parameters.i_max),
        "i_min": -float(parameters.i_max),
        "effort_limit_nm": FIXED_EFFORT_LIMIT_NM,
    }
    by_joint = controller.get("wheel_velocity_pid_by_joint")
    if not isinstance(by_joint, dict):
        by_joint = controller.get("per_wheel")
    if isinstance(by_joint, dict) and by_joint:
        for joint_name in WHEEL_JOINTS:
            facts = by_joint.get(joint_name)
            if not isinstance(facts, dict):
                check["missing"].append(f"{source}.wheel_velocity_pid_by_joint.{joint_name}")
                continue
            for parameter_name, expected in expected_values.items():
                location = (
                    f"{source}.wheel_velocity_pid_by_joint."
                    f"{joint_name}.{parameter_name}"
                )
                actual = facts.get(parameter_name)
                check["checked"].append(
                    {
                        "parameter": parameter_name,
                        "location": location,
                        "actual": actual,
                        "expected": expected,
                    }
                )
                if not _matches(actual, expected):
                    check["mismatches"].append(
                        _mismatch(parameter_name, expected, actual, location)
                    )
    else:
        facts = controller.get("wheel_velocity_pid")
        if not isinstance(facts, dict):
            check["status"] = "missing"
            check["missing"].append(f"{source}.wheel_velocity_pid")
            return check, f"runtime_metadata.{scope}.pid_contract_unverifiable"
        for parameter_name, expected in expected_values.items():
            location = f"{source}.wheel_velocity_pid.{parameter_name}"
            actual = facts.get(parameter_name)
            check["checked"].append(
                {
                    "parameter": parameter_name,
                    "location": location,
                    "actual": actual,
                    "expected": expected,
                }
            )
            if not _matches(actual, expected):
                check["mismatches"].append(
                    _mismatch(parameter_name, expected, actual, location)
                )

    if check["mismatches"]:
        check["status"] = "mismatch"
        return check, f"runtime_metadata.{scope}.pid_contract_mismatch"
    if check["missing"]:
        check["status"] = "partial"
        return check, f"runtime_metadata.{scope}.pid_contract_unverifiable"
    check["status"] = "match"
    return check, None


def _mujoco_model(metadata: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    runtime_configuration = metadata.get("runtime_configuration")
    if isinstance(runtime_configuration, dict):
        model = runtime_configuration.get("mujoco_model")
        if isinstance(model, dict):
            return model, "runtime_configuration.mujoco_model"
    model = metadata.get("mujoco_model")
    if isinstance(model, dict):
        return model, "mujoco_model"
    return None, None


def _ground_contact_friction(
    metadata: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    runtime_configuration = metadata.get("runtime_configuration")
    if isinstance(runtime_configuration, dict):
        friction = runtime_configuration.get("ground_contact_friction")
        if isinstance(friction, dict):
            return friction, "runtime_configuration.ground_contact_friction"
    friction = metadata.get("ground_contact_friction")
    if isinstance(friction, dict):
        return friction, "ground_contact_friction"
    return None, None


def _validate_model_friction(
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    model, source = _mujoco_model(metadata)
    if model is None:
        return None
    pairs = model.get("contact_pair_friction_by_name")
    check: dict[str, Any] = {
        "source": source,
        "status": "unknown",
        "checked": [],
        "missing": [],
        "mismatches": [],
        "complete_match": False,
    }
    if not isinstance(pairs, dict):
        check["status"] = "missing"
        check["missing"].append(f"{source}.contact_pair_friction_by_name")
        return check

    for pair_name in WHEEL_CONTACT_PAIRS:
        entry = pairs.get(pair_name)
        if not isinstance(entry, dict):
            check["missing"].append(f"{source}.contact_pair_friction_by_name.{pair_name}")
            continue
        friction = entry.get("friction")
        if (
            not isinstance(friction, list)
            or len(friction) <= PRIMARY_FRICTION_INDEX
        ):
            check["missing"].append(
                f"{source}.contact_pair_friction_by_name.{pair_name}."
                f"friction[{PRIMARY_FRICTION_INDEX}]"
            )
            continue
        location = (
            f"{source}.contact_pair_friction_by_name."
            f"{pair_name}.friction[{PRIMARY_FRICTION_INDEX}]"
        )
        actual = friction[PRIMARY_FRICTION_INDEX]
        check["checked"].append(
            {
                "parameter": "mu_high",
                "location": location,
                "actual": actual,
                "expected": FIXED_MU_HIGH,
            }
        )
        if not _matches(actual, FIXED_MU_HIGH):
            check["mismatches"].append(
                _mismatch("mu_high", FIXED_MU_HIGH, actual, location)
            )

    if check["mismatches"]:
        check["status"] = "mismatch"
    elif check["missing"]:
        check["status"] = "partial"
    else:
        check["status"] = "match"
        check["complete_match"] = True
    return check


def _validate_observed_friction(
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    friction_report, source = _ground_contact_friction(metadata)
    if friction_report is None:
        return None
    pairs = friction_report.get("pairs")
    check: dict[str, Any] = {
        "source": source,
        "status": "unknown",
        "checked": [],
        "missing": [],
        "mismatches": [],
        "complete_match": False,
    }
    if not isinstance(pairs, dict):
        check["status"] = "missing"
        check["missing"].append(f"{source}.pairs")
        return check

    by_geom: dict[str, dict[str, Any]] = {}
    for pair_key, entry in pairs.items():
        if not isinstance(entry, dict):
            continue
        other_geom = entry.get("other_geom")
        if isinstance(other_geom, str) and other_geom in WHEEL_GROUND_GEOMS:
            by_geom[other_geom] = entry | {"_pair_key": pair_key}

    for geom_name in WHEEL_GROUND_GEOMS:
        entry = by_geom.get(geom_name)
        if not isinstance(entry, dict):
            check["missing"].append(f"{source}.pairs.{geom_name}")
            continue
        vectors = entry.get("friction_vectors")
        if not isinstance(vectors, list) or not vectors:
            check["missing"].append(f"{source}.pairs.{entry['_pair_key']}.friction_vectors")
            continue
        for vector_index, vector in enumerate(vectors):
            if (
                not isinstance(vector, list)
                or len(vector) <= PRIMARY_FRICTION_INDEX
            ):
                check["missing"].append(
                    f"{source}.pairs.{entry['_pair_key']}."
                    f"friction_vectors[{vector_index}]"
                    f"[{PRIMARY_FRICTION_INDEX}]"
                )
                continue
            location = (
                f"{source}.pairs.{entry['_pair_key']}."
                f"friction_vectors[{vector_index}][{PRIMARY_FRICTION_INDEX}]"
            )
            actual = vector[PRIMARY_FRICTION_INDEX]
            check["checked"].append(
                {
                    "parameter": "mu_high",
                    "location": location,
                    "actual": actual,
                    "expected": FIXED_MU_HIGH,
                }
            )
            if not _matches(actual, FIXED_MU_HIGH):
                check["mismatches"].append(
                    _mismatch("mu_high", FIXED_MU_HIGH, actual, location)
                )

    if check["mismatches"]:
        check["status"] = "mismatch"
    elif check["missing"]:
        check["status"] = "partial"
    else:
        check["status"] = "match"
        check["complete_match"] = True
    return check


def validate_traction_runtime_metadata(
    parameters: EvaluationParameters,
    run_dir: Path,
    expected_axis: str | None = None,
    expected_condition_id: str | None = None,
    expected_speed_mps: float | None = None,
    scope: str | None = None,
) -> RuntimeMetadataValidation:
    if expected_condition_id is not None:
        condition_axis_value = condition_axis(expected_condition_id)
        condition_speed_value = condition_speed_mps(expected_condition_id)
        if expected_axis is not None and expected_axis != condition_axis_value:
            raise ValueError("expected axis and condition disagree")
        expected_axis = condition_axis_value
        if expected_speed_mps is not None and not math.isclose(
            float(expected_speed_mps),
            condition_speed_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("expected speed and condition disagree")
        expected_speed_mps = condition_speed_value
    scope = scope or (
        f"traction_{expected_condition_id}"
        if expected_condition_id else f"traction_{expected_axis or 'unknown'}"
    )
    metadata_path = run_dir / "run_metadata.json"
    metadata, error = _read_json(metadata_path)
    if metadata is None:
        check = {
            "kind": "traction_runtime_metadata",
            "scope": scope,
            "metadata_path": str(metadata_path),
            "status": "missing",
            "error": error,
        }
        return RuntimeMetadataValidation(
            valid=False,
            reject_reasons=[f"runtime_metadata.{scope}.metadata_missing"],
            checks=[check],
        )

    actual_axis = str(metadata.get("axis", ""))
    axis = expected_axis or actual_axis
    scope = scope if expected_axis else f"traction_{axis or 'unknown'}"
    expected_names = {
        "tendon_name": f"traction_cable_{axis}",
        "sensor_name": f"traction_cable_{axis}_limit_force",
        "attach_site_name": f"traction_attach_{axis}",
        "anchor_site_name": f"traction_anchor_{axis}",
    }
    expected_attach = [-0.32, 0.0, 0.217] if axis == "x" else [0.0, -0.11, 0.217]
    expected_anchor = [-2.0, 0.0, 0.217] if axis == "x" else [0.0, -2.0, 0.217]
    check: dict[str, Any] = {
        "kind": "traction_tendon_contract",
        "scope": scope,
        "metadata_path": str(metadata_path),
        "status": "match",
        "mismatches": [],
    }
    reject_reasons: list[str] = []

    def reject(kind: str, detail: str) -> None:
        reason = f"runtime_metadata.{scope}.{kind}"
        if reason not in reject_reasons:
            reject_reasons.append(reason)
        check["mismatches"].append(detail)

    if (
        metadata.get("experiment_type") != "traction_tendon_pull"
        or metadata.get("measurement_method") != "tendon_limit"
        or metadata.get("schema_version") != 1
    ):
        reject("measurement_contract_mismatch", "method/schema")
    if axis not in {"x", "y"} or actual_axis != axis:
        reject("axis_mismatch", f"expected {axis}, got {actual_axis}")
    if expected_speed_mps is not None:
        profile = metadata.get("command_profile")
        actual_speed = (
            profile.get("target_axis_velocity_mps")
            if isinstance(profile, dict) else None
        )
        if not _matches(actual_speed, expected_speed_mps):
            reject(
                "target_speed_mismatch",
                f"expected {expected_speed_mps}, got {actual_speed}",
            )
    for field, expected in expected_names.items():
        if metadata.get(field) != expected:
            reject("site_mismatch" if "site" in field else "tendon_mismatch", field)

    for field, expected in (
        ("attach_initial_world_m", expected_attach),
        ("anchor_initial_world_m", expected_anchor),
    ):
        actual = metadata.get(field)
        if not isinstance(actual, list) or len(actual) != 3 or any(
            not _matches(actual[index], expected[index]) for index in range(3)
        ):
            reject("site_mismatch", field)

    length0 = _float_or_none(metadata.get("tendon_length0_m"))
    upper = _float_or_none(metadata.get("tendon_upper_limit_m"))
    slack = _float_or_none(metadata.get("configured_slack_m"))
    if (
        length0 is None
        or upper is None
        or slack is None
        or not math.isclose(upper - length0, slack, abs_tol=1e-8, rel_tol=0.0)
        or not math.isclose(slack, 0.001, abs_tol=1e-8, rel_tol=0.0)
        or not _matches(metadata.get("tendon_lower_limit_m"), 0.0)
        or any(
            not _matches(metadata.get(field), 0.0)
            for field in (
                "tendon_stiffness_n_per_m",
                "tendon_damping_n_s_per_m",
                "tendon_frictionloss_n",
            )
        )
    ):
        reject("tendon_mismatch", "length/range/passive parameters")

    solver = metadata.get("solver")
    if (
        not isinstance(solver, dict)
        or solver.get("cone") != "ELLIPTIC"
        or not str(solver.get("name", "")).strip()
        or (_float_or_none(solver.get("iterations")) or 0.0) <= 0.0
        or (_float_or_none(solver.get("timestep_s")) or 0.0) <= 0.0
    ):
        reject("solver_mismatch", "solver/cone/iterations/timestep")

    controller = metadata.get("controller")
    if (
        not isinstance(controller, dict)
        or not _matches(controller.get("wheel_velocity_kp"), parameters.kp)
        or not _matches(controller.get("wheel_velocity_ki"), parameters.ki)
        or not _matches(controller.get("wheel_integral_limit"), parameters.i_max)
        or not _matches(
            controller.get("wheel_effort_limit_nm"), FIXED_EFFORT_LIMIT_NM
        )
    ):
        reject("pid_contract_mismatch", "wheel velocity PID")

    controller_check, controller_reason = _validate_controller_parameters(
        metadata,
        metadata_path,
        parameters,
        scope,
    )
    if controller_reason:
        reject("pid_contract_mismatch", controller_reason)

    friction = metadata.get("wheel_ground_friction")
    if (
        not isinstance(friction, dict)
        or not _matches(friction.get("rolling"), FIXED_MU_LOW)
        or not _matches(friction.get("lateral"), FIXED_MU_HIGH)
    ):
        reject("fixed_friction_mismatch", "wheel_ground_friction")
    if not isinstance(friction, dict) or friction.get("solref_friction") != [-1.0, -1.0]:
        reject("friction_solver_mismatch", "solref_friction")

    expected_up_pose = [0.0, -math.pi / 2.0, 0.0, -math.pi / 2.0, 0.0, 0.0]
    declared_preset = metadata.get("expected_ur5_preset")
    declared_pose = metadata.get("expected_ur5_pose_rad")
    runtime_ur5 = metadata.get("ur5_runtime_initial")
    runtime_pose = (
        runtime_ur5.get("positions_rad")
        if isinstance(runtime_ur5, dict)
        else None
    )
    runtime_velocity = (
        runtime_ur5.get("velocities_rad_s")
        if isinstance(runtime_ur5, dict)
        else None
    )
    if (
        declared_preset != "up"
        or not isinstance(declared_pose, list)
        or len(declared_pose) != 6
        or any(_float_or_none(value) is None for value in declared_pose)
        or any(
            not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9)
            for actual, expected in zip(declared_pose, expected_up_pose)
        )
    ):
        reject("ur5_pose_contract_mismatch", "expected_ur5_preset/pose")
    if not isinstance(runtime_pose, list) or len(runtime_pose) != 6 or any(
        _float_or_none(value) is None for value in runtime_pose
    ):
        reject("ur5_pose_unverifiable", "ur5_runtime_initial.positions_rad")
    elif any(
        not math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=0.01
        )
        for actual, expected in zip(runtime_pose, expected_up_pose)
    ):
        reject("ur5_pose_mismatch", "expected up preset")
    if not isinstance(runtime_velocity, list) or len(runtime_velocity) != 6 or any(
        _float_or_none(value) is None for value in runtime_velocity
    ):
        reject("ur5_velocity_unverifiable", "ur5_runtime_initial.velocities_rad_s")
    elif any(abs(float(value)) > 0.01 for value in runtime_velocity):
        reject("ur5_velocity_mismatch", "expected settled up preset")

    runtime = metadata.get("runtime_parameters")
    if not isinstance(runtime, dict):
        reject("observer_config_mismatch", "runtime_parameters")
    else:
        for field, expected in expected_names.items():
            runtime_key = f"traction_tendon.{field}"
            if runtime.get(runtime_key) != expected:
                reject("observer_config_mismatch", runtime_key)
    check["status"] = "match" if not reject_reasons else "mismatch"

    return RuntimeMetadataValidation(
        valid=not reject_reasons,
        reject_reasons=reject_reasons,
        checks=[check, controller_check],
    )


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]] | None, str | None]:
    if not path.exists():
        return None, f"missing file: {path}"
    try:
        with path.open("r", encoding="utf-8", newline="") as file_obj:
            return list(csv.DictReader(file_obj)), None
    except OSError as exc:
        return None, f"cannot read CSV: {path}: {exc}"


def _frf_run_metadata_candidates(row: dict[str, str], summary_path: Path) -> list[Path]:
    candidates: list[Path] = []
    source_run_dir = row.get("source_run_dir")
    if source_run_dir:
        candidates.append(Path(source_run_dir) / "run_metadata.json")

    source_experiment_dir = row.get("source_experiment_dir")
    run_id = row.get("run_id")
    if source_experiment_dir and run_id:
        candidates.append(Path(source_experiment_dir) / "runs" / run_id / "run_metadata.json")

    source_experiment = row.get("source_experiment")
    if source_experiment and run_id:
        summary_root = summary_path.parent.parent
        candidates.append(summary_root / source_experiment / "runs" / run_id / "run_metadata.json")

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _resolve_existing_metadata_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def validate_frf_runtime_metadata(
    parameters: EvaluationParameters,
    sim_summary_csv: Path,
) -> RuntimeMetadataValidation:
    rows, error = _read_csv_rows(sim_summary_csv)
    if rows is None:
        check = {
            "kind": "frf_runtime_metadata",
            "scope": "frf",
            "summary_path": str(sim_summary_csv),
            "status": "missing",
            "error": error,
        }
        return RuntimeMetadataValidation(
            valid=False,
            reject_reasons=["runtime_metadata.frf.summary_missing"],
            checks=[check],
        )

    checks: list[dict[str, Any]] = []
    reject_reasons: list[str] = []
    checked_paths: set[str] = set()
    missing_rows: list[dict[str, Any]] = []

    for row in rows:
        candidates = _frf_run_metadata_candidates(row, sim_summary_csv)
        metadata_path = _resolve_existing_metadata_path(candidates)
        if metadata_path is None:
            missing_rows.append(
                {
                    "run_id": row.get("run_id"),
                    "source_experiment": row.get("source_experiment"),
                    "candidates": [str(candidate) for candidate in candidates],
                }
            )
            continue

        resolved_key = str(metadata_path.resolve())
        if resolved_key in checked_paths:
            continue
        checked_paths.add(resolved_key)

        metadata, metadata_error = _read_json(metadata_path)
        if metadata is None:
            checks.append(
                {
                    "kind": "wheel_velocity_pid",
                    "scope": "frf",
                    "metadata_path": str(metadata_path),
                    "status": "missing",
                    "error": metadata_error,
                }
            )
            reject_reasons.append("runtime_metadata.frf.metadata_missing")
            continue

        check, reason = _validate_controller_parameters(
            metadata,
            metadata_path,
            parameters,
            scope="frf",
        )
        check["frequency_hz"] = metadata.get("frequency_hz")
        checks.append(check)
        if reason:
            reject_reasons.append(reason)

    if missing_rows:
        checks.append(
            {
                "kind": "frf_runtime_metadata",
                "scope": "frf",
                "summary_path": str(sim_summary_csv),
                "status": "missing",
                "missing_rows": missing_rows,
            }
        )
        reject_reasons.append("runtime_metadata.frf.metadata_missing")

    if not checked_paths and not missing_rows:
        checks.append(
            {
                "kind": "frf_runtime_metadata",
                "scope": "frf",
                "summary_path": str(sim_summary_csv),
                "status": "missing",
                "error": "frequency summary has no metadata-bearing rows",
            }
        )
        reject_reasons.append("runtime_metadata.frf.metadata_missing")

    reject_reasons = list(dict.fromkeys(reject_reasons))
    return RuntimeMetadataValidation(
        valid=not reject_reasons,
        reject_reasons=reject_reasons,
        checks=checks,
    )


def validate_runtime_metadata(
    parameters: EvaluationParameters,
    traction_run_dir: Path | None,
    frf_sim_summary: Path | None,
) -> RuntimeMetadataValidation:
    combined_checks: list[dict[str, Any]] = []
    combined_reasons: list[str] = []

    if traction_run_dir is not None:
        traction = validate_traction_runtime_metadata(parameters, traction_run_dir)
        combined_checks.extend(traction.checks)
        combined_reasons.extend(traction.reject_reasons)

    if frf_sim_summary is not None:
        frf = validate_frf_runtime_metadata(parameters, frf_sim_summary)
        combined_checks.extend(frf.checks)
        combined_reasons.extend(frf.reject_reasons)

    combined_reasons = list(dict.fromkeys(combined_reasons))
    return RuntimeMetadataValidation(
        valid=not combined_reasons,
        reject_reasons=combined_reasons,
        checks=combined_checks,
    )


def validate_runtime_metadata_multi(
    parameters: EvaluationParameters,
    traction_run_dirs: dict[str, Path | None],
    frf_sim_summary: Path | None,
) -> RuntimeMetadataValidation:
    combined_checks: list[dict[str, Any]] = []
    combined_reasons: list[str] = []

    for condition_id in TRACTION_CONDITION_IDS:
        run_dir = traction_run_dirs.get(condition_id)
        if run_dir is None:
            continue
        traction = validate_traction_runtime_metadata(
            parameters,
            run_dir,
            expected_condition_id=condition_id,
            scope=f"traction_{condition_id}",
        )
        combined_checks.extend(traction.checks)
        combined_reasons.extend(traction.reject_reasons)

    if frf_sim_summary is not None:
        frf = validate_frf_runtime_metadata(parameters, frf_sim_summary)
        combined_checks.extend(frf.checks)
        combined_reasons.extend(frf.reject_reasons)

    combined_reasons = list(dict.fromkeys(combined_reasons))
    return RuntimeMetadataValidation(
        valid=not combined_reasons,
        reject_reasons=combined_reasons,
        checks=combined_checks,
    )
