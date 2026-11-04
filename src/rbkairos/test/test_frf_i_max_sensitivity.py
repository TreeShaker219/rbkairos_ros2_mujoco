import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PACKAGE_ROOT / "scripts"
RECORDER = SCRIPTS / "frequency_response_recorder.py"
PILOT = SCRIPTS / "frf_i_max_sensitivity.py"
RUNNER = SCRIPTS / "run_frf_i_max_sensitivity_pilot.py"
CMAKE = PACKAGE_ROOT / "CMakeLists.txt"
sys.path.insert(0, str(SCRIPTS))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_pid_message(module):
    names = list(module.WHEEL_DOF_NAMES)
    targets = [1.0, 2.0, 3.0, 4.0]
    actual = [0.5, 1.5, 2.5, 3.5]
    error = [0.5] * 4
    p_term = [25.0] * 4
    i_term = [0.5] * 4
    raw = [25.5] * 4
    return SimpleNamespace(
        velocity_pid_diagnostic_schema_version=2,
        velocity_pid_joint_names=names,
        velocity_pid_target=targets,
        velocity_pid_actual=actual,
        velocity_pid_error=error,
        velocity_pid_p_term=p_term,
        velocity_pid_i_term=i_term,
        velocity_pid_candidate_i_term=[0.6] * 4,
        velocity_pid_peak_abs_candidate_i_term=[0.6] * 4,
        velocity_pid_d_term=[0.0] * 4,
        velocity_pid_raw_output=raw,
        velocity_pid_effort_clamped_output=raw,
        velocity_pid_i_clamp_active=[True] * 4,
        velocity_pid_effort_clamp_active=[False] * 4,
        velocity_pid_i_clamp_ever=[True] * 4,
        velocity_pid_effort_clamp_ever=[False] * 4,
        velocity_pid_control_update_count=[42] * 4,
        velocity_pid_kp=[50.0] * 4,
        velocity_pid_ki=[1.0] * 4,
        velocity_pid_kd=[0.0] * 4,
        velocity_pid_i_max=[0.5] * 4,
        velocity_pid_i_min=[-0.5] * 4,
        velocity_pid_antiwindup=[True] * 4,
        velocity_pid_effort_limit=[100.0] * 4,
        dof_names=["world_joint#0", *names],
        qfrc_applied=[0.0, *raw],
    )


def test_frf_recorder_decodes_and_records_read_only_pid_diagnostics():
    module = load_module(RECORDER, "frequency_response_recorder")

    decoded = module.decode_velocity_pid_diagnostics(make_pid_message(module))

    assert list(decoded) == list(module.WHEEL_DOF_NAMES)
    assert decoded["left_front_wheel_joint"]["i_term"] == pytest.approx(0.5)
    assert decoded["left_front_wheel_joint"]["candidate_i_term"] == pytest.approx(0.6)
    assert decoded["left_front_wheel_joint"]["i_clamp_ever"] is True
    assert decoded["right_back_wheel_joint"]["qfrc_applied"] == pytest.approx(25.5)
    source = RECORDER.read_text(encoding="utf-8")
    assert '"wheel_velocity_pid.csv"' in source
    assert "decode_velocity_pid_diagnostics(msg)" in source
    args, _ = module.parse_args([])
    assert args.write_velocity_pid_diagnostics is False
    assert "--write-velocity-pid-diagnostics" in RUNNER.read_text(encoding="utf-8")


def test_frf_recorder_rejects_pid_output_that_differs_from_qfrc_applied():
    module = load_module(RECORDER, "frequency_response_recorder_invalid")
    message = make_pid_message(module)
    message.qfrc_applied[1] = 24.0

    with pytest.raises(module.PidDiagnosticContractError, match="qfrc_applied"):
        module.decode_velocity_pid_diagnostics(message)


def test_pilot_plan_freezes_one_run_per_full_formal_frf_cell():
    module = load_module(PILOT, "frf_i_max_sensitivity")

    plan = module.default_pilot_plan()

    assert plan["fixed"] == {
        "mu_low": 0.01,
        "mu_high": 0.7,
        "effort_limit_nm": 100.0,
        "kd": 0.0,
        "ur5_preset": "up",
        "amplitude_mps": 0.10,
    }
    assert [candidate["i_max"] for candidate in plan["candidates"]] == [0.5, 1.5, 3.0]
    assert plan["frequencies_hz"] == [0.2, 0.3, 0.5, 1.0, 1.8, 2.8, 4.0]
    assert len(plan["runs"]) == 21
    assert {run["replicate"] for run in plan["runs"]} == {1}


def write_pid_fixture(path: Path, module, *, i_max: float, clamped: bool):
    fields = list(module.PID_CSV_FIELDS)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for sample_time in (1.0, 2.0, 3.0):
            for wheel in module.WHEEL_DOF_NAMES:
                writer.writerow(
                    {
                        "t_experiment": sample_time,
                        "header_stamp_s": 100.0 + sample_time,
                        "joint_name": wheel,
                        "velocity_target": 1.0,
                        "velocity_actual": 0.5,
                        "velocity_error": 0.5,
                        "p_term": 25.0,
                        "i_term": i_max if clamped else 0.25,
                        "d_term": 0.0,
                        "pid_raw_output": 25.0 + (i_max if clamped else 0.25),
                        "effort_clamped_output": 25.0 + (i_max if clamped else 0.25),
                        "qfrc_applied": 25.0 + (i_max if clamped else 0.25),
                        "i_clamp_active": int(clamped),
                        "effort_clamp_active": 0,
                        "kp": 50.0,
                        "ki": 1.0,
                        "kd": 0.0,
                        "i_max": i_max,
                        "i_min": -i_max,
                        "antiwindup": 1,
                        "effort_limit_nm": 100.0,
                    }
                )


def test_pid_window_summary_uses_fft_window_and_reports_clamp_by_wheel(tmp_path):
    module = load_module(PILOT, "frf_i_max_sensitivity_summary")
    path = tmp_path / "wheel_velocity_pid.csv"
    write_pid_fixture(path, module, i_max=0.5, clamped=True)

    summary = module.summarize_pid_window(
        path,
        analysis_start_s=1.5,
        analysis_end_s=3.0,
        expected={"kp": 50.0, "ki": 1.0, "kd": 0.0, "i_max": 0.5},
    )

    assert summary["sample_rows"] == 8
    assert summary["all_wheel_i_clamp_ratio"] == pytest.approx(1.0)
    assert summary["any_wheel_effort_clamp_ratio"] == pytest.approx(0.0)
    assert summary["max_abs_i_term_nm"] == pytest.approx(0.5)
    assert set(summary["per_wheel"]) == set(module.WHEEL_DOF_NAMES)


def test_aggregate_report_uses_same_gain_only_j_frf_and_writes_artifacts(tmp_path):
    module = load_module(PILOT, "frf_i_max_sensitivity_report")
    rows = []
    for i_max, gain in ((0.5, -1.0), (1.5, -0.5), (3.0, 0.0)):
        rows.append(
            {
                "candidate_id": f"imax{i_max}",
                "kp": 50.0,
                "ki": 1.0,
                "i_max": i_max,
                "frequency_hz": 1.0,
                "gain_db_sim": gain,
                "gain_db_real": -1.0,
                "phase_deg_sim": -20.0,
                "valid": True,
                "i_clamp_any": i_max == 0.5,
                "all_wheel_i_clamp_ratio": 1.0 if i_max == 0.5 else 0.0,
                "any_wheel_effort_clamp_ratio": 0.0,
                "max_abs_i_term_nm": min(i_max, 0.5),
            }
        )

    report = module.write_pilot_reports(rows, tmp_path)

    by_i_max = {item["i_max"]: item for item in report["candidates"]}
    assert by_i_max[0.5]["J_frf"] == pytest.approx(0.0)
    assert by_i_max[1.5]["J_frf"] == pytest.approx(0.25)
    assert by_i_max[3.0]["J_frf"] == pytest.approx(1.0)
    assert report["per_frequency"][0]["gain_span_db"] == pytest.approx(1.0)
    assert (tmp_path / "frf_i_max_sensitivity_report.json").is_file()
    assert (tmp_path / "frf_i_max_sensitivity_summary.csv").is_file()
    assert (tmp_path / "frf_i_max_gain_overlay.svg").is_file()
    json.loads((tmp_path / "frf_i_max_sensitivity_report.json").read_text())


def test_pilot_scripts_are_installed():
    cmake = CMAKE.read_text(encoding="utf-8")

    assert "scripts/run_frf_i_max_sensitivity_pilot.py" in cmake
    assert "scripts/frf_i_max_sensitivity.py" in cmake
    assert RUNNER.is_file()
