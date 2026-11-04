import importlib.util
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PACKAGE_ROOT / "scripts"
PILOT = SCRIPTS / "traction_tendon_i_max_pilot.py"
RUNNER = SCRIPTS / "run_traction_tendon_i_max_pilot.py"
CMAKE = PACKAGE_ROOT / "CMakeLists.txt"
sys.path.insert(0, str(SCRIPTS))


def load_pilot_module():
    spec = importlib.util.spec_from_file_location("traction_tendon_i_max_pilot", PILOT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runner_module():
    spec = importlib.util.spec_from_file_location("run_traction_tendon_i_max_pilot", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_plan_freezes_protocol_before_runtime():
    module = load_pilot_module()

    plan = module.default_pilot_plan()

    assert plan["schema_version"] == 1
    assert plan["status"] == "planned_before_runtime"
    fixed = plan["fixed"]
    fixed_keys = (
        "mu_low", "mu_high", "effort_limit_nm", "kd", "ur5_preset"
    )
    assert {key: fixed[key] for key in fixed_keys} == {
        "mu_low": 0.01,
        "mu_high": 0.7,
        "effort_limit_nm": 100.0,
        "kd": 0.0,
        "ur5_preset": "up",
    }
    assert fixed["ur5_pose_rad"] == pytest.approx(module.UP_POSE_RAD)
    assert [candidate["i_max"] for candidate in plan["candidates"]] == [0.5, 1.5, 3.0]
    assert {candidate["kp"] for candidate in plan["candidates"]} == {50.0}
    assert {candidate["ki"] for candidate in plan["candidates"]} == {1.0}
    assert len(plan["runs"]) == 36
    assert {(run["axis"], run["speed_mps"]) for run in plan["runs"]} == {
        (axis, speed) for axis in ("x", "y") for speed in (0.01, 0.05, 0.10)
    }
    assert {run["replicate"] for run in plan["runs"]} == {1, 2}
    assert min(run["constant_duration_s"] for run in plan["runs"]) > 4.0
    assert plan["analysis"]["uses_fixed_final_two_seconds"] is False


def test_candidate_assets_change_only_pilot_pid_pose_and_fixed_friction(tmp_path):
    module = load_pilot_module()
    candidate = module.default_pilot_plan()["candidates"][1]
    assets = module.prepare_candidate_assets(
        candidate,
        tmp_path / "assets",
        source_urdf=PACKAGE_ROOT / "urdf" / "rbkairos.urdf",
        source_mjcf=PACKAGE_ROOT / "mujoco" / "rbkairos.xml",
        source_scenes={
            axis: PACKAGE_ROOT / "mujoco" / f"rbkairos_scene_tendon_test_{axis}.xml"
            for axis in ("x", "y")
        },
    )

    urdf_root = ET.parse(assets["urdf_path"]).getroot()
    wheel_names = set(module.WHEEL_JOINTS)
    wheels = [
        joint for joint in urdf_root.findall(".//ros2_control/joint")
        if joint.get("name") in wheel_names
    ]
    assert len(wheels) == 4
    for joint in wheels:
        params = {item.get("name"): float(item.text) for item in joint.findall("param")}
        assert params["velocity_kp"] == 50.0
        assert params["velocity_ki"] == 1.0
        assert params["velocity_kd"] == 0.0
        assert params["velocity_i_max"] == 1.5
        assert params["velocity_i_min"] == -1.5
        physical = urdf_root.find(f"./joint[@name='{joint.get('name')}']/limit")
        assert float(physical.get("effort")) == 100.0

    arm_initial = module.read_control_initial_pose(urdf_root, module.UR5_JOINTS)
    assert arm_initial == pytest.approx(module.UP_POSE_RAD)

    mjcf_root = ET.parse(assets["mjcf_path"]).getroot()
    friction = [
        [float(value) for value in pair.get("friction").split()[:2]]
        for pair in mjcf_root.findall(".//contact/pair")
        if pair.get("name") in module.WHEEL_CONTACT_PAIRS
    ]
    assert friction == [[0.01, 0.7]] * 4
    for scene_path in assets["scene_paths"].values():
        include = ET.parse(scene_path).getroot().find("./include")
        assert Path(include.get("file")).resolve() == Path(assets["mjcf_path"]).resolve()


def synthetic_rows(module, end_s=14.0, effort_clamp=False):
    state = []
    odom = []
    pid = []
    step = 0.1
    index = 0
    while index * step <= end_s + 1e-9:
        t = round(index * step, 10)
        # Deliberately looks low-CV at four seconds but does not reach its plateau
        # until eight seconds.
        tension = 100.0 + 5.0 * min(t, 8.0)
        state.append({
            "t_experiment": t,
            "tension_n": tension,
            "limit_active_flag": 1.0,
            "extension_over_upper_limit_m": 0.001,
        })
        odom.append({
            "t_experiment": t,
            "vx": 0.0,
            "vy": 0.0,
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
        })
        for wheel in module.WHEEL_JOINTS:
            i_term = min(0.25 * t, 2.0)
            pid.append({
                "t_experiment": t,
                "joint_name": wheel,
                "i_term": i_term,
                "i_clamp_active": 1.0 if t >= 8.0 else 0.0,
                "effort_clamp_active": 1.0 if effort_clamp and t >= 9.0 else 0.0,
                "qfrc_applied": 12.0 + i_term,
            })
        index += 1
    return state, odom, pid


def test_retrospective_plateau_is_not_fixed_four_second_tail():
    module = load_pilot_module()
    state, odom, pid = synthetic_rows(module)

    result = module.analyze_time_series(
        state,
        odom,
        pid,
        axis="x",
        constant_start_s=0.0,
        constant_end_s=14.0,
    )

    assert result["converged"] is True
    assert result["time_to_stable_s"] >= 8.0
    assert result["plateau_force_n"] == pytest.approx(140.0)
    assert result["all_wheels_i_clamped"] is True
    assert result["effort_clamp_any"] is False
    assert result["per_wheel"][module.WHEEL_JOINTS[0]]["time_to_i_clamp_s"] == pytest.approx(8.0)


def test_truncated_four_second_record_is_not_called_steady():
    module = load_pilot_module()
    state, odom, pid = synthetic_rows(module, end_s=4.0)

    result = module.analyze_time_series(
        state,
        odom,
        pid,
        axis="x",
        constant_start_s=0.0,
        constant_end_s=4.0,
    )

    assert result["converged"] is False
    assert "not_converged" in result["reject_reasons"]
    assert result["plateau_force_n"] is None


def test_effort_clamp_rejects_instead_of_redefining_the_metric():
    module = load_pilot_module()
    state, odom, pid = synthetic_rows(module, effort_clamp=True)

    result = module.analyze_time_series(
        state,
        odom,
        pid,
        axis="x",
        constant_start_s=0.0,
        constant_end_s=14.0,
    )

    assert result["effort_clamp_any"] is True
    assert result["valid"] is False
    assert "effort_clamp_active" in result["reject_reasons"]


def test_aggregate_requires_repeatable_monotonic_and_sufficient_effect(tmp_path):
    module = load_pilot_module()
    rows = []
    for axis in ("x", "y"):
        for speed in (0.01, 0.05, 0.10):
            for i_max, force in ((0.5, 100.0), (1.5, 110.0), (3.0, 125.0)):
                for replicate, multiplier in ((1, 1.0), (2, 1.005)):
                    rows.append({
                        "axis": axis,
                        "speed_mps": speed,
                        "candidate_id": f"kp50_ki1_imax{i_max}",
                        "i_max": i_max,
                        "replicate": replicate,
                        "plateau_force_n": force * multiplier,
                        "valid": True,
                        "converged": True,
                        "effort_clamp_any": False,
                    })

    aggregate = module.aggregate_pilot_results(rows)

    assert aggregate["valid_cell_count"] == 6
    assert aggregate["all_cells_repeatable"] is True
    assert aggregate["all_cells_monotonic"] is True
    assert aggregate["all_cells_sufficient"] is True
    assert aggregate["i_max_identifiable"] is True
    reports = module.write_matrix_reports(rows, tmp_path)
    assert Path(reports["force_plot"]).is_file()
    assert Path(reports["convergence_plot"]).is_file()
    assert Path(reports["i_clamp_plot"]).is_file()


def test_runtime_contract_rejects_old_friction_wrong_pid_and_wrong_arm_pose():
    module = load_pilot_module()
    candidate = module.default_pilot_plan()["candidates"][0]
    run_spec = next(
        run for run in module.default_pilot_plan()["runs"]
        if run["candidate_id"] == candidate["candidate_id"]
    )
    per_wheel = {
        wheel: {
            "kp": candidate["kp"],
            "ki": candidate["ki"],
            "kd": 0.0,
            "i_max": candidate["i_max"],
            "i_min": candidate["i_min"],
            "effort_limit_nm": 100.0,
        }
        for wheel in module.WHEEL_JOINTS
    }
    metadata = {
        "experiment_type": "traction_tendon_pull",
        "measurement_method": "tendon_limit",
        "schema_version": 1,
        "axis": run_spec["axis"],
        "wheel_ground_friction": {"rolling": 0.01, "lateral": 0.7},
        "command_profile": {
            "target_axis_velocity_mps": run_spec["speed_mps"],
            "constant_start_s": 1.5,
            "constant_end_s": 1.5 + run_spec["constant_duration_s"],
        },
        "velocity_pid_runtime": {"per_wheel": per_wheel},
        "expected_ur5_preset": "up",
        "ur5_runtime_initial": {
            "positions_rad": list(module.UP_POSE_RAD),
            "velocities_rad_s": [0.0] * 6,
        },
        "run_status": {"completed": True, "valid": True},
    }

    module.validate_runtime_contract(metadata, run_spec, candidate)
    metadata["wheel_ground_friction"]["lateral"] = 0.65
    with pytest.raises(module.PilotInputError, match="mu_high"):
        module.validate_runtime_contract(metadata, run_spec, candidate)
    metadata["wheel_ground_friction"]["lateral"] = 0.7
    metadata["velocity_pid_runtime"]["per_wheel"][module.WHEEL_JOINTS[0]]["i_max"] = 9.0
    with pytest.raises(module.PilotInputError, match="i_max"):
        module.validate_runtime_contract(metadata, run_spec, candidate)


def test_pilot_scripts_are_installed_and_do_not_rewire_formal_evaluation():
    cmake = CMAKE.read_text(encoding="utf-8")
    runner_source = RUNNER.read_text(encoding="utf-8")

    assert "scripts/run_traction_tendon_i_max_pilot.py" in cmake
    assert "scripts/traction_tendon_i_max_pilot.py" in cmake
    assert "evaluation.parameters" not in runner_source
    assert "formal evaluation" not in runner_source.lower()


def test_controller_readiness_parser_ignores_ros2_color_codes():
    module = load_runner_module()
    output = (
        "\x1b[92mjoint_state_broadcaster        \x1b[0m type \x1b[92mactive\x1b[0m\n"
        "\x1b[92mur5_joint_trajectory_controller\x1b[0m type \x1b[92mactive\x1b[0m\n"
        "\x1b[92mmecanum_base_controller\x1b[0m type \x1b[92mactive\x1b[0m\n"
    )

    assert module.active_controllers(output) == {
        "joint_state_broadcaster",
        "ur5_joint_trajectory_controller",
        "mecanum_base_controller",
    }


def test_pilot_flag_parser_accepts_exact_csv_boolean_encodings():
    module = load_pilot_module()

    assert module._flag("1.0", "flag") is True
    assert module._flag("0.0", "flag") is False
    with pytest.raises(module.PilotInputError):
        module._flag("0.5", "flag")
