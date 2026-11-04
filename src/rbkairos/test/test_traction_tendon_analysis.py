import csv
import json
import math
from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PACKAGE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import traction_tendon_analysis as analysis  # noqa: E402


def canonical_metadata(axis="x"):
    names = analysis.AXIS_CONTRACTS[axis]
    attach = [-0.32, 0.0, 0.217] if axis == "x" else [0.0, -0.11, 0.217]
    anchor = [-2.0, 0.0, 0.217] if axis == "x" else [0.0, -2.0, 0.217]
    length0 = 1.68 if axis == "x" else 1.89
    return {
        "experiment_type": "traction_tendon_pull",
        "measurement_method": "tendon_limit",
        "schema_version": 1,
        "state_schema": "traction_tendon_state_v1",
        "summary_schema_version": 2,
        "axis": axis,
        "model_file": f"mujoco/rbkairos_scene_tendon_test_{axis}.xml",
        "tendon_name": names["tendon_name"],
        "sensor_name": names["sensor_name"],
        "attach_site_name": names["attach_site_name"],
        "anchor_site_name": names["anchor_site_name"],
        "attach_initial_world_m": attach,
        "anchor_initial_world_m": anchor,
        "tendon_length0_m": length0,
        "tendon_lower_limit_m": 0.0,
        "tendon_upper_limit_m": length0 + 0.001,
        "configured_slack_m": 0.001,
        "tendon_stiffness_n_per_m": 0.0,
        "tendon_damping_n_s_per_m": 0.0,
        "tendon_frictionloss_n": 0.0,
        "solver": {
            "name": "Newton",
            "cone": "ELLIPTIC",
            "iterations": 50,
            "timestep_s": 0.001,
            "tendon_solref_limit": [0.02, 1.0],
            "tendon_solimp_limit": [0.9, 0.95, 0.001, 0.5, 2.0],
        },
        "controller": {
            "wheel_velocity_kp": 50.0,
            "wheel_velocity_ki": 0.1,
            "wheel_effort_limit_nm": 100.0,
            "wheel_integral_limit": 1.0,
        },
        "wheel_ground_friction": {
            "rolling": 0.01,
            "lateral": 0.7,
            "solref_friction": [-1.0, -1.0],
        },
        "ur5_joint_pose_rad": [0.0] * 6,
        "command_profile": {
            "warmup_s": 1.0,
            "ramp_s": 0.5,
            "constant_s": 4.0,
            "min_observation_s": 2.0,
            "max_wait_s": 60.0,
            "post_stop_s": 0.5,
            "constant_start_s": 1.5,
            "constant_end_s": 5.5,
            "target_axis_velocity_mps": 0.1,
            "ramp_shape": "cosine",
        },
        "steady_state_contract": {
            "method": "adaptive_sustained_windows_v1",
            "window_s": 2.0,
            "window_step_s": 0.25,
            "min_observation_s": 2.0,
            "required_stable_windows": 4,
            "max_wait_s": 60.0,
            "min_tendon_active_ratio": 0.95,
            "max_force_cv": 0.05,
            "max_abs_force_slope_n_s": 0.5,
            "max_constraint_extension_m": 0.005,
            "max_abs_base_velocity_mps": 0.005,
            "max_abs_i_term_slope_nm_s": 0.005,
            "min_i_clamp_ratio": 0.95,
        },
        "force_sign_convention": (
            "tension_n_is_nonnegative; force_xyz_n_is_force_on_attach_toward_anchor_in_world"
        ),
        "observer_publish_rate_hz_expected": 10.0,
        "sampling_contract": {
            "state_expected_rate_hz": 10.0,
            "odom_expected_rate_hz": 10.0,
            "min_sample_fraction": 0.8,
            "max_state_gap_s": 0.15,
            "max_odom_gap_s": 0.2,
        },
        "run_status": {
            "completed": True,
            "valid": True,
            "reject_reason": "",
        },
    }


def state_row(t, axis="x", tension=100.0, active=1.0, extension=0.001):
    length0 = 1.68 if axis == "x" else 1.89
    force_x = -tension if axis == "x" else 0.0
    force_y = -tension if axis == "y" else 0.0
    attach_x = -0.32 if axis == "x" else 0.0
    attach_y = 0.0 if axis == "x" else -0.11
    anchor_x = -2.0 if axis == "x" else 0.0
    anchor_y = 0.0 if axis == "x" else -2.0
    values = [
        t,
        length0 + 0.001 + extension,
        length0,
        0.0,
        length0 + 0.001,
        extension,
        tension,
        active,
        force_x,
        force_y,
        0.0,
        attach_x,
        attach_y,
        0.217,
        anchor_x,
        anchor_y,
        0.217,
    ]
    state = dict(zip(analysis.STATE_FIELDS, values))
    return {
        "t_experiment": t,
        "command_phase": "constant" if 1.5 <= t <= 5.5 else "ramp",
        **state,
    }


def odom_row(t, axis="x", speed=0.001):
    return {
        "t_experiment": t,
        "header_stamp_s": t,
        "x": 0.002 * t if axis == "x" else 0.0001 * t,
        "y": 0.002 * t if axis == "y" else 0.0001 * t,
        "z": 0.127,
        "roll": 0.001,
        "pitch": -0.001,
        "yaw": 0.002,
        "vx": speed if axis == "x" else 0.0001,
        "vy": speed if axis == "y" else 0.0001,
        "vz": 0.0,
        "wx": 0.0,
        "wy": 0.0,
        "wz": 0.0,
    }


def write_csv(path, rows):
    rows = list(rows)
    assert rows
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_run(root, axis="x", *, metadata=None, state_rows=None, odom_rows=None):
    run_dir = root / f"run_{axis}"
    run_dir.mkdir(parents=True)
    metadata = canonical_metadata(axis) if metadata is None else metadata
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    times = [round(index * 0.1, 6) for index in range(61)]
    if state_rows is None:
        state_rows = [
            state_row(
                value,
                axis,
                tension=100.0 + (index % 2),
                active=0.0 if math.isclose(value, 3.5) else 1.0,
            )
            for index, value in enumerate(times)
        ]
    if odom_rows is None:
        odom_rows = [odom_row(value, axis) for value in times]
    write_csv(run_dir / "traction_tendon_state.csv", state_rows)
    write_csv(run_dir / "odom.csv", odom_rows)
    write_csv(
        run_dir / "wheel_qfrc_summary.csv",
        [
            {
                "t_experiment": value,
                "max_abs_qfrc_applied": 80.0,
                "left_front_wheel_joint_qfrc_applied": 80.0,
                "right_front_wheel_joint_qfrc_applied": 79.0,
                "left_back_wheel_joint_qfrc_applied": 78.0,
                "right_back_wheel_joint_qfrc_applied": 77.0,
            }
            for value in times
        ],
    )
    write_csv(
        run_dir / "wheel_velocity_pid.csv",
        [
            {
                "t_experiment": value,
                "joint_name": wheel,
                "i_term": 1.0,
                "i_clamp_active": True,
                "effort_clamp_active": False,
                "qfrc_applied": 80.0,
            }
            for value in times
            for wheel in analysis.WHEEL_DOF_ORDER
        ],
    )
    return run_dir


def test_v1_state_contract_decodes_only_the_exact_layout():
    assert analysis.STATE_SCHEMA_VERSION == 1
    assert len(analysis.STATE_FIELDS) == 17
    values = list(range(17))
    values[7] = 1

    decoded = analysis.decode_state_message(analysis.STATE_LAYOUT_LABEL, values)

    assert list(decoded) == list(analysis.STATE_FIELDS)
    assert decoded["tension_n"] == 6.0
    with pytest.raises(analysis.AnalysisInputError, match="layout label"):
        analysis.decode_state_message("traction_tendon_state_v2:wrong", values)
    with pytest.raises(analysis.AnalysisInputError, match="17 fields"):
        analysis.decode_state_message(analysis.STATE_LAYOUT_LABEL, values[:-1])
    with pytest.raises(analysis.AnalysisInputError, match="non-finite"):
        analysis.decode_state_message(
            analysis.STATE_LAYOUT_LABEL,
            values[:6] + [math.nan] + values[7:],
        )


def test_metadata_validation_accepts_canonical_tendon_run_and_rejects_legacy_spring():
    analysis.validate_metadata(canonical_metadata("x"), requested_axis="x")

    legacy = canonical_metadata("x")
    legacy.update(
        experiment_type="virtual_spring_pull",
        measurement_method="virtual_spring",
    )
    with pytest.raises(analysis.AnalysisInputError, match="experiment_type"):
        analysis.validate_metadata(legacy)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.update(schema_version=2), "schema_version"),
        (lambda data: data.update(axis="y"), "tendon_name"),
        (lambda data: data.update(attach_site_name="spring_attach"), "attach_site_name"),
        (lambda data: data.update(configured_slack_m=0.002), "slack"),
        (lambda data: data.update(tendon_stiffness_n_per_m=1.0), "stiffness"),
        (lambda data: data["solver"].update(cone="PYRAMIDAL"), "solver.cone"),
        (
            lambda data: data["wheel_ground_friction"].update(
                solref_friction=[0.0, 0.0]
            ),
            "solref_friction",
        ),
    ],
)
def test_metadata_validation_rejects_contract_mismatches(mutation, message):
    metadata = canonical_metadata("x")
    mutation(metadata)
    with pytest.raises(analysis.AnalysisInputError, match=message):
        analysis.validate_metadata(metadata)


def test_single_run_uses_the_sustained_adaptive_plateau_not_a_fixed_tail(tmp_path):
    run_dir = write_run(tmp_path)

    summary = analysis.analyze_traction_tendon_run(run_dir)

    assert summary["axis"] == "x"
    assert summary["measurement_method"] == "tendon_limit"
    assert summary["steady_method"] == "adaptive_sustained_windows_v1"
    assert summary["steady_converged"] is True
    assert summary["stable_window_count"] >= 4
    assert summary["steady_tow_force_n"] == pytest.approx(100.5)
    assert summary["steady_tow_force_mean_n"] == pytest.approx(100.5)
    assert summary["steady_tow_force_std_n"] == pytest.approx(0.5)
    assert summary["steady_tow_force_cv"] == pytest.approx(0.5 / 100.5)
    assert summary["tendon_active_ratio"] == pytest.approx(40.0 / 41.0)
    assert summary["peak_tow_force_n"] == pytest.approx(101.0)
    assert summary["steady_axis_force_n"] == pytest.approx(100.5)
    assert summary["steady_cross_axis_force_n"] == pytest.approx(0.0)
    assert summary["max_constraint_extension_m"] == pytest.approx(0.001)
    assert summary["steady_base_velocity_axis_mps"] == pytest.approx(0.001)
    assert summary["wheel_qfrc_applied_max"] == pytest.approx(80.0)
    assert summary["wheel_saturation_ratio"] == pytest.approx(0.8)
    assert summary["limit_mode"] == "not_effort_limited"
    assert summary["valid"] is True
    assert summary["reject_reason"] == ""

    assert (run_dir / "traction_tendon_summary.json").is_file()
    assert (run_dir / "traction_tendon_summary.csv").is_file()
    for plot_name in analysis.RUN_PLOT_FILENAMES:
        assert (run_dir / plot_name).is_file()


def test_single_run_prefers_runtime_pid_effort_limit_over_urdf_declaration(tmp_path):
    metadata = canonical_metadata("x")
    metadata["velocity_pid_runtime"] = {
        "schema_version": 1,
        "source_topic": "/mujoco_dof_state",
        "per_wheel": {
            name: {
                "kp": 50.0,
                "ki": 0.1,
                "kd": 0.0,
                "i_max": 1.0,
                "i_min": -1.0,
                "antiwindup": True,
                "effort_limit_nm": 50.0,
            }
            for name in (
                "left_front_wheel_joint",
                "right_front_wheel_joint",
                "left_back_wheel_joint",
                "right_back_wheel_joint",
            )
        },
    }
    run_dir = write_run(tmp_path, metadata=metadata)

    summary = analysis.analyze_traction_tendon_run(run_dir)

    assert summary["wheel_saturation_ratio"] == pytest.approx(1.6)
    assert summary["limit_mode"] == "wheel_effort_limited"


def test_runtime_pid_metadata_rejects_nonuniform_effort_limits():
    metadata = canonical_metadata("x")

    def config(limit):
        return {
            "kp": 50.0,
            "ki": 0.1,
            "kd": 0.0,
            "i_max": 1.0,
            "i_min": -1.0,
            "antiwindup": True,
            "effort_limit_nm": limit,
        }

    metadata["velocity_pid_runtime"] = {
        "schema_version": 1,
        "source_topic": "/mujoco_dof_state",
        "per_wheel": {
            "left_front_wheel_joint": config(100.0),
            "right_front_wheel_joint": config(100.0),
            "left_back_wheel_joint": config(100.0),
            "right_back_wheel_joint": config(90.0),
        },
    }

    with pytest.raises(analysis.AnalysisInputError, match="not uniform"):
        analysis.validate_metadata(metadata)


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("inactive", "active_ratio"),
        ("high_cv", "force_cv"),
        ("overextended", "constraint_extension"),
        ("moving", "base_velocity"),
    ],
)
def test_validity_gates_reject_bad_steady_behavior(tmp_path, case, reason):
    times = [round(index * 0.1, 6) for index in range(61)]
    rows = []
    odom = []
    for index, value in enumerate(times):
        active = 0.0 if case == "inactive" and value >= 3.5 and index % 2 == 0 else 1.0
        tension = 60.0 if case == "high_cv" and index % 2 == 0 else 100.0
        extension = 0.006 if case == "overextended" else 0.001
        rows.append(state_row(value, tension=tension, active=active, extension=extension))
        odom.append(odom_row(value, speed=0.02 if case == "moving" else 0.001))
    run_dir = write_run(tmp_path, state_rows=rows, odom_rows=odom)

    summary = analysis.analyze_traction_tendon_run(run_dir)

    assert summary["valid"] is False
    assert reason in summary["reject_reason"]


@pytest.mark.parametrize(
    ("case", "reason"),
    [("slow_integral", "i_term_slope"), ("effort_clamp", "effort_clamp")],
)
def test_formal_analyzer_uses_pid_stability_and_effort_clamp(tmp_path, case, reason):
    run_dir = write_run(tmp_path)
    times = [round(index * 0.1, 6) for index in range(61)]
    write_csv(
        run_dir / "wheel_velocity_pid.csv",
        [
            {
                "t_experiment": value,
                "joint_name": wheel,
                "i_term": 0.02 * value if case == "slow_integral" else 1.0,
                "i_clamp_active": 0.0 if case == "slow_integral" else 1.0,
                "effort_clamp_active": (
                    1.0 if case == "effort_clamp" and value >= 3.0 else 0.0
                ),
                "qfrc_applied": 80.0,
            }
            for value in times
            for wheel in analysis.WHEEL_DOF_ORDER
        ],
    )

    summary = analysis.analyze_traction_tendon_run(run_dir)

    assert summary["valid"] is False
    assert reason in summary["reject_reason"]
    assert summary["steady_tow_force_n"] is None or math.isnan(
        summary["steady_tow_force_n"]
    )


@pytest.mark.parametrize("case", ["empty", "nonfinite", "time_reversal"])
def test_analysis_fails_fast_for_invalid_core_samples(tmp_path, case):
    times = [round(index * 0.1, 6) for index in range(61)]
    rows = [state_row(value) for value in times]
    if case == "empty":
        rows = rows[:20]
    elif case == "nonfinite":
        rows[40]["tension_n"] = "nan"
    else:
        rows[40]["t_experiment"] = rows[39]["t_experiment"] - 0.01
    run_dir = write_run(tmp_path, state_rows=rows)

    expected = "non-finite" if case == "nonfinite" else case.replace("_", " ")
    with pytest.raises(analysis.AnalysisInputError, match=expected):
        analysis.analyze_traction_tendon_run(run_dir)


def test_batch_discovers_only_tendon_runs_and_reports_descriptive_groups(tmp_path):
    first = write_run(tmp_path / "tendon_a", axis="x")
    second = write_run(tmp_path / "tendon_b", axis="x")
    legacy = tmp_path / "legacy_spring"
    legacy.mkdir()
    (legacy / "run_metadata.json").write_text(
        json.dumps({"experiment_type": "virtual_spring_pull"}), encoding="utf-8"
    )
    (legacy / "virtual_spring_state.csv").write_text(
        "t_experiment,tension_n\n0,1\n", encoding="utf-8"
    )

    discovered = analysis.discover_run_dirs(tmp_path)
    summaries = analysis.analyze_traction_tendon_batch(tmp_path)

    assert discovered == sorted([first, second])
    assert len(summaries) == 2
    assert (tmp_path / "traction_tendon_batch_summary.csv").is_file()
    assert (tmp_path / "traction_tendon_repeatability.svg").is_file()
    report = json.loads(
        (tmp_path / "traction_tendon_batch_report.json").read_text(encoding="utf-8")
    )
    assert report["run_count"] == 2
    assert report["groups"]["x"]["relative_range"] == pytest.approx(0.0)
    assert "accepted" not in report["groups"]["x"]
    assert "repeatability_within_3_percent" not in report["groups"]["x"]


def test_batch_reports_validity_without_repeatability_or_solver_thresholds(tmp_path):
    for index in range(3):
        metadata = canonical_metadata("x")
        metadata["solver_variant"] = "baseline"
        if index == 0:
            metadata["run_status"] = {
                "completed": True,
                "valid": False,
                "reject_reason": "fixture_preflight_failure",
            }
        write_run(tmp_path / f"baseline_{index}", axis="x", metadata=metadata)

    variant_metadata = canonical_metadata("x")
    variant_metadata["solver_variant"] = "iterations100"
    write_run(tmp_path / "iterations100", axis="x", metadata=variant_metadata)

    analysis.analyze_traction_tendon_batch(tmp_path)

    report = json.loads(
        (tmp_path / "traction_tendon_batch_report.json").read_text(encoding="utf-8")
    )
    baseline = report["groups"]["x"]
    comparison = report["solver_sensitivity"]["x"][0]
    assert baseline["run_count"] == 3
    assert baseline["valid_run_count"] == 2
    assert comparison["run_valid"] is True
    assert "within_3_percent" not in comparison
    assert "accepted" not in comparison


def test_sampling_gap_and_claimed_tendon_corruption_fail_fast(tmp_path):
    times = [round(index * 0.1, 6) for index in range(61) if index != 40]
    run_dir = write_run(
        tmp_path / "gapped",
        state_rows=[state_row(value) for value in times],
    )
    with pytest.raises(analysis.AnalysisInputError, match="state sample gap"):
        analysis.analyze_traction_tendon_run(run_dir)

    broken = tmp_path / "batch" / "broken_tendon"
    broken.mkdir(parents=True)
    metadata = canonical_metadata("x")
    metadata["sensor_name"] = "wrong"
    (broken / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (broken / "traction_tendon_state.csv").write_text("broken\n", encoding="utf-8")
    with pytest.raises(analysis.AnalysisInputError, match="malformed Tendon run"):
        analysis.discover_run_dirs(tmp_path / "batch")
