import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))
RECORDER = PACKAGE_ROOT / "scripts" / "traction_tendon_experiment.py"
CMAKE = PACKAGE_ROOT / "CMakeLists.txt"


def load_recorder_module():
    spec = importlib.util.spec_from_file_location("traction_tendon_experiment", RECORDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_command_profile_waits_in_constant_until_adaptive_monitor_finishes():
    module = load_recorder_module()

    assert module.command_profile_at(0.5, 1.0, 0.5, 0.5) == ("warmup", 0.0, False)
    phase, scale, done = module.command_profile_at(1.25, 1.0, 0.5, 0.5)
    assert phase == "ramp"
    assert scale == pytest.approx(0.5)
    assert done is False
    assert module.command_profile_at(20.0, 1.0, 0.5, 0.5) == (
        "constant",
        1.0,
        False,
    )
    assert module.command_profile_at(9.75, 1.0, 0.5, 0.5, constant_end_s=9.5) == (
        "post_stop",
        0.0,
        False,
    )
    assert module.command_profile_at(10.0, 1.0, 0.5, 0.5, constant_end_s=9.5) == (
        "complete",
        0.0,
        True,
    )


def test_axis_defaults_match_the_two_c1_scenes():
    module = load_recorder_module()

    assert module.axis_defaults("x") == {
        "model_file": "mujoco/rbkairos_scene_tendon_test_x.xml",
        "tendon_name": "traction_cable_x",
        "sensor_name": "traction_cable_x_limit_force",
        "attach_site_name": "traction_attach_x",
        "anchor_site_name": "traction_anchor_x",
    }
    assert module.axis_defaults("y")["attach_site_name"] == "traction_attach_y"


def test_model_metadata_records_tendon_contact_regularization():
    module = load_recorder_module()
    names = module.axis_defaults("x")
    model_path = PACKAGE_ROOT / names.pop("model_file")

    facts = module.inspect_mujoco_model(model_path, names)

    assert facts["solver"]["cone"] == "ELLIPTIC"
    assert facts["wheel_ground_friction"]["solref_friction"] == [-1.0, -1.0]
    assert facts["wheel_ground_friction"]["rolling"] == pytest.approx(0.01)
    assert facts["wheel_ground_friction"]["lateral"] == pytest.approx(0.7)


def test_new_recorder_has_no_virtual_spring_measurement_semantics_and_is_installed():
    source = RECORDER.read_text(encoding="utf-8")
    cmake = CMAKE.read_text(encoding="utf-8")

    forbidden = (
        "virtual_spring_state.csv",
        "spring stiffness",
        "stop_on_stall",
        "stall_rollback",
        "spring_saturated",
    )
    for token in forbidden:
        assert token not in source
    assert "traction_tendon_state.csv" in source
    assert "traction_tendon_analysis" in source
    assert "scripts/traction_tendon_experiment.py" in cmake
    assert "scripts/analyze_traction_tendon.py" in cmake


def test_cli_rejects_nonpositive_phase_durations_and_cross_axis_command():
    module = load_recorder_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--max-wait-duration", "0"])
    with pytest.raises(SystemExit):
        module.parse_args([
            "--min-observation-duration", "10", "--max-wait-duration", "5"
        ])
    with pytest.raises(SystemExit):
        module.parse_args(["--axis", "x", "--velocity-y", "0.1"])


def test_cli_exposes_the_complete_adaptive_steady_contract():
    module = load_recorder_module()

    args, _ = module.parse_args([])

    assert args.min_observation_duration == pytest.approx(2.0)
    assert args.max_wait_duration == pytest.approx(60.0)
    assert args.steady_window == pytest.approx(2.0)
    assert args.steady_window_step == pytest.approx(0.25)
    assert args.required_stable_windows == 4
    assert args.max_force_slope == pytest.approx(0.5)
    assert args.max_i_term_slope == pytest.approx(0.005)
    assert args.min_i_clamp_ratio == pytest.approx(0.95)
    assert not hasattr(args, "constant_duration")

    source = RECORDER.read_text(encoding="utf-8")
    assert "AdaptiveSteadyMonitor" in source
    assert '"not_converged"' in source


def make_pid_message(module):
    names = list(module.WHEEL_DOF_ORDER)
    targets = [1.0, 2.0, 3.0, 4.0]
    actual = [0.5, 1.5, 2.5, 3.5]
    error = [0.5] * 4
    p_term = [25.0] * 4
    i_term = [0.2] * 4
    d_term = [0.0] * 4
    raw = [25.2] * 4
    clamped = list(raw)
    return SimpleNamespace(
        velocity_pid_diagnostic_schema_version=2,
        velocity_pid_joint_names=names,
        velocity_pid_target=targets,
        velocity_pid_actual=actual,
        velocity_pid_error=error,
        velocity_pid_p_term=p_term,
        velocity_pid_i_term=i_term,
        velocity_pid_candidate_i_term=i_term,
        velocity_pid_peak_abs_candidate_i_term=i_term,
        velocity_pid_d_term=d_term,
        velocity_pid_raw_output=raw,
        velocity_pid_effort_clamped_output=clamped,
        velocity_pid_i_clamp_active=[False] * 4,
        velocity_pid_effort_clamp_active=[False] * 4,
        velocity_pid_i_clamp_ever=[False] * 4,
        velocity_pid_effort_clamp_ever=[False] * 4,
        velocity_pid_control_update_count=[42] * 4,
        velocity_pid_kp=[50.0] * 4,
        velocity_pid_ki=[0.1] * 4,
        velocity_pid_kd=[0.0] * 4,
        velocity_pid_i_max=[1.0] * 4,
        velocity_pid_i_min=[-1.0] * 4,
        velocity_pid_antiwindup=[True] * 4,
        velocity_pid_effort_limit=[100.0] * 4,
        dof_names=["world_joint#0", *names],
        qfrc_applied=[0.0, *clamped],
    )


def test_pid_diagnostic_decoder_validates_four_wheels_and_numeric_identities():
    module = load_recorder_module()
    message = make_pid_message(module)

    decoded = module.decode_velocity_pid_diagnostics(message)

    assert list(decoded) == list(module.WHEEL_DOF_ORDER)
    assert decoded["left_front_wheel_joint"]["p_term"] == pytest.approx(25.0)
    assert decoded["left_front_wheel_joint"]["control_update_count"] == 42
    assert decoded["right_back_wheel_joint"]["effort_clamped_output"] == pytest.approx(
        25.2
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.velocity_pid_joint_names.pop(),
            "exactly the four wheel joints",
        ),
        (
            lambda value: value.velocity_pid_p_term.pop(),
            "velocity_pid_p_term",
        ),
        (
            lambda value: value.velocity_pid_i_term.__setitem__(0, float("nan")),
            "non-finite",
        ),
        (
            lambda value: value.velocity_pid_raw_output.__setitem__(0, 24.0),
            r"P\+I\+D",
        ),
        (
            lambda value: value.qfrc_applied.__setitem__(1, 24.0),
            "qfrc_applied",
        ),
    ],
)
def test_pid_diagnostic_decoder_rejects_malformed_or_inconsistent_samples(mutate, message):
    module = load_recorder_module()
    value = make_pid_message(module)
    mutate(value)

    with pytest.raises(module.PidDiagnosticContractError, match=message):
        module.decode_velocity_pid_diagnostics(value)


def test_runtime_pid_metadata_comes_from_diagnostics_not_urdf_defaults():
    module = load_recorder_module()
    decoded = module.decode_velocity_pid_diagnostics(make_pid_message(module))

    metadata = module.runtime_velocity_pid_metadata(decoded, "/mujoco_dof_state")

    assert metadata["schema_version"] == 1
    assert metadata["source_topic"] == "/mujoco_dof_state"
    assert metadata["per_wheel"]["left_front_wheel_joint"] == {
        "kp": 50.0,
        "ki": 0.1,
        "kd": 0.0,
        "i_max": 1.0,
        "i_min": -1.0,
        "antiwindup": True,
        "effort_limit_nm": 100.0,
    }


def test_recorder_declares_pid_csv_and_runtime_metadata_contract():
    source = RECORDER.read_text(encoding="utf-8")

    assert '"wheel_velocity_pid.csv"' in source
    assert '"velocity_pid_runtime"' in source
    assert "decode_velocity_pid_diagnostics(message)" in source


def test_recorder_can_require_and_record_the_runtime_up_pose():
    module = load_recorder_module()
    message = SimpleNamespace(
        dof_names=[*module.UR5_JOINT_ORDER],
        qpos=[0.0, -module.math.pi / 2.0, 0.0, -module.math.pi / 2.0, 0.0, 0.0],
        qvel=[0.0] * 6,
    )

    state = module.decode_ur5_state(message)
    error = module.validate_ur5_pose(
        state,
        preset="up",
        position_tolerance_rad=0.01,
        velocity_tolerance_rad_s=0.01,
    )

    assert error is None
    assert state["positions_rad"] == pytest.approx(module.UR5_PRESETS["up"])
    source = RECORDER.read_text(encoding="utf-8")
    assert '"ur5_state.csv"' in source
    assert '"expected_ur5_preset"' in source
