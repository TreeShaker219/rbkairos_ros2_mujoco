from pathlib import Path


WORKSPACE_SRC = Path(__file__).resolve().parents[2]
MSG = (
    WORKSPACE_SRC
    / "mujoco_ros2_control"
    / "mujoco_ros2_control_msgs"
    / "msg"
    / "MujocoDofState.msg"
)
SYSTEM_HEADER = (
    WORKSPACE_SRC
    / "mujoco_ros2_control"
    / "mujoco_ros2_control"
    / "include"
    / "mujoco_ros2_control"
    / "mujoco_system.hpp"
)
SYSTEM_SOURCE = (
    WORKSPACE_SRC
    / "mujoco_ros2_control"
    / "mujoco_ros2_control"
    / "src"
    / "mujoco_system.cpp"
)
PUBLISHER_SOURCE = (
    WORKSPACE_SRC
    / "mujoco_ros2_control"
    / "mujoco_ros2_control"
    / "src"
    / "mujoco_ros2_control.cpp"
)


EXPECTED_MESSAGE_FIELDS = (
    "uint32 velocity_pid_diagnostic_schema_version",
    "string[] velocity_pid_joint_names",
    "float64[] velocity_pid_target",
    "float64[] velocity_pid_actual",
    "float64[] velocity_pid_error",
    "float64[] velocity_pid_p_term",
    "float64[] velocity_pid_i_term",
    "float64[] velocity_pid_candidate_i_term",
    "float64[] velocity_pid_peak_abs_candidate_i_term",
    "float64[] velocity_pid_d_term",
    "float64[] velocity_pid_raw_output",
    "float64[] velocity_pid_effort_clamped_output",
    "bool[] velocity_pid_i_clamp_active",
    "bool[] velocity_pid_effort_clamp_active",
    "bool[] velocity_pid_i_clamp_ever",
    "bool[] velocity_pid_effort_clamp_ever",
    "uint64[] velocity_pid_control_update_count",
    "float64[] velocity_pid_kp",
    "float64[] velocity_pid_ki",
    "float64[] velocity_pid_kd",
    "float64[] velocity_pid_i_max",
    "float64[] velocity_pid_i_min",
    "bool[] velocity_pid_antiwindup",
    "float64[] velocity_pid_effort_limit",
)


def test_dof_message_has_versioned_velocity_pid_diagnostic_contract():
    source = MSG.read_text(encoding="utf-8")

    for field in EXPECTED_MESSAGE_FIELDS:
        assert field in source


def test_system_collects_diagnostics_without_replacing_the_pid_or_effort_path():
    header = SYSTEM_HEADER.read_text(encoding="utf-8")
    source = SYSTEM_SOURCE.read_text(encoding="utf-8")

    assert "get_velocity_pid_diagnostics() const override" in header
    assert "compute_velocity_pid_diagnostic(" in source
    assert "velocity_pid.computeCommand(" not in source
    assert "qfrc_applied[joint_state.mj_vel_adr] =" in source
    assert ".effort_clamped_output" in source


def test_dof_publisher_exposes_every_runtime_diagnostic_field():
    source = PUBLISHER_SOURCE.read_text(encoding="utf-8")

    assert "kVelocityPidDiagnosticSchemaVersion" in source
    assert "kVelocityPidDiagnosticSchemaVersion = 2" in (
        WORKSPACE_SRC
        / "mujoco_ros2_control"
        / "mujoco_ros2_control"
        / "include"
        / "mujoco_ros2_control"
        / "velocity_pid_diagnostics.hpp"
    ).read_text(encoding="utf-8")
    for field in EXPECTED_MESSAGE_FIELDS[1:]:
        field_name = field.split()[-1].removesuffix("[]")
        assert f"msg.{field_name}" in source
