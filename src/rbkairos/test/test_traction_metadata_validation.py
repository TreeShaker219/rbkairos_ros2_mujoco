import copy
import json
from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from evaluation.metadata_validation import validate_traction_runtime_metadata  # noqa: E402
from evaluation.parameters import EvaluationParameters  # noqa: E402
from evaluation.traction_conditions import (  # noqa: E402
    TRACTION_CONDITION_IDS,
    condition_axis,
    condition_speed_mps,
)


def canonical_metadata(condition_id="x_v0p10"):
    axis = condition_axis(condition_id)
    speed = condition_speed_mps(condition_id)
    attach = [-0.32, 0.0, 0.217] if axis == "x" else [0.0, -0.11, 0.217]
    anchor = [-2.0, 0.0, 0.217] if axis == "x" else [0.0, -2.0, 0.217]
    length0 = 1.68 if axis == "x" else 1.89
    return {
        "experiment_type": "traction_tendon_pull",
        "measurement_method": "tendon_limit",
        "schema_version": 1,
        "axis": axis,
        "model_file": f"mujoco/rbkairos_scene_tendon_test_{axis}.xml",
        "tendon_name": f"traction_cable_{axis}",
        "sensor_name": f"traction_cable_{axis}_limit_force",
        "attach_site_name": f"traction_attach_{axis}",
        "anchor_site_name": f"traction_anchor_{axis}",
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
            "name": "NEWTON",
            "cone": "ELLIPTIC",
            "iterations": 50,
            "timestep_s": 0.001,
        },
        "controller": {
            "wheel_velocity_kp": 50.0,
            "wheel_velocity_ki": 1.0,
            "wheel_effort_limit_nm": 100.0,
            "wheel_integral_limit": 3.0,
        },
        "velocity_pid_runtime": {
            "schema_version": 1,
            "per_wheel": {
                joint: {
                    "kp": 50.0,
                    "ki": 1.0,
                    "kd": 0.0,
                    "i_max": 3.0,
                    "i_min": -3.0,
                    "effort_limit_nm": 100.0,
                }
                for joint in (
                    "left_front_wheel_joint",
                    "right_front_wheel_joint",
                    "left_back_wheel_joint",
                    "right_back_wheel_joint",
                )
            },
        },
        "wheel_ground_friction": {
            "rolling": 0.01,
            "lateral": 0.7,
            "solref_friction": [-1.0, -1.0],
        },
        "ur5_joint_pose_rad": [0.0, -1.57079632679, 0.0, -1.57079632679, 0.0, 0.0],
        "expected_ur5_preset": "up",
        "expected_ur5_pose_rad": [
            0.0,
            -1.57079632679,
            0.0,
            -1.57079632679,
            0.0,
            0.0,
        ],
        "ur5_runtime_initial": {
            "positions_rad": [
                0.0,
                -1.57079632679,
                0.0,
                -1.57079632679,
                0.0,
                0.0,
            ],
            "velocities_rad_s": [0.0] * 6,
        },
        "command_profile": {"target_axis_velocity_mps": speed},
        "runtime_parameters": {
            "model_file": f"mujoco/rbkairos_scene_tendon_test_{axis}.xml",
            "traction_tendon.tendon_name": f"traction_cable_{axis}",
            "traction_tendon.sensor_name": f"traction_cable_{axis}_limit_force",
            "traction_tendon.attach_site_name": f"traction_attach_{axis}",
            "traction_tendon.anchor_site_name": f"traction_anchor_{axis}",
        },
    }


def write_run(tmp_path, metadata):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return run_dir


@pytest.mark.parametrize("condition_id", TRACTION_CONDITION_IDS)
def test_canonical_traction_metadata_matches_candidate(condition_id, tmp_path):
    result = validate_traction_runtime_metadata(
        EvaluationParameters(kp=50.0, ki=1.0, i_max=3.0),
        write_run(tmp_path, canonical_metadata(condition_id)),
        expected_condition_id=condition_id,
    )

    assert result.valid is True
    assert result.reject_reasons == []


def test_runtime_up_pose_is_authoritative_over_model_qpos0(tmp_path):
    metadata = canonical_metadata("x_v0p10")
    metadata["ur5_joint_pose_rad"] = [0.0] * 6

    result = validate_traction_runtime_metadata(
        EvaluationParameters(kp=50.0, ki=1.0, i_max=3.0),
        write_run(tmp_path, metadata),
        expected_condition_id="x_v0p10",
    )

    assert result.valid is True
    assert result.reject_reasons == []


def test_runtime_non_up_pose_is_rejected_even_when_model_qpos0_is_up(tmp_path):
    metadata = canonical_metadata("x_v0p10")
    metadata["ur5_runtime_initial"]["positions_rad"] = [0.0] * 6

    result = validate_traction_runtime_metadata(
        EvaluationParameters(kp=50.0, ki=1.0, i_max=3.0),
        write_run(tmp_path, metadata),
        expected_condition_id="x_v0p10",
    )

    assert result.valid is False
    assert any("ur5_pose_mismatch" in item for item in result.reject_reasons)


@pytest.mark.parametrize(
    "mutation, reason",
    [
        (
            lambda data: data.update(
                experiment_type="virtual_spring_pull",
                measurement_method="virtual_spring",
            ),
            "measurement_contract_mismatch",
        ),
        (lambda data: data.update(axis="y"), "axis_mismatch"),
        (lambda data: data.update(attach_site_name="spring_attach"), "site_mismatch"),
        (lambda data: data.update(configured_slack_m=0.002), "tendon_mismatch"),
        (lambda data: data["solver"].update(cone="PYRAMIDAL"), "solver_mismatch"),
        (
            lambda data: data["velocity_pid_runtime"]["per_wheel"][
                "left_front_wheel_joint"
            ].update(i_max=9.0),
            "pid_contract_mismatch",
        ),
        (
            lambda data: data["wheel_ground_friction"].update(lateral=0.4),
            "fixed_friction_mismatch",
        ),
        (
            lambda data: data["command_profile"].update(
                target_axis_velocity_mps=0.05
            ),
            "target_speed_mismatch",
        ),
    ],
)
def test_traction_metadata_rejects_contract_mismatches(mutation, reason, tmp_path):
    metadata = copy.deepcopy(canonical_metadata("x_v0p10"))
    mutation(metadata)

    result = validate_traction_runtime_metadata(
        EvaluationParameters(kp=50.0, ki=1.0, i_max=3.0),
        write_run(tmp_path, metadata),
        expected_condition_id="x_v0p10",
    )

    assert result.valid is False
    assert any(reason in item for item in result.reject_reasons)
