#!/usr/bin/env python3
"""Prepare, execute, resume, and analyze the isolated FRF i_max sensitivity pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from evaluation.parameters import EvaluationParameters  # noqa: E402
from evaluation.runtime_assets import create_runtime_assets  # noqa: E402
from frf_i_max_sensitivity import (  # noqa: E402
    PilotInputError,
    build_candidate_rows,
    default_pilot_plan,
    real_gain_map,
    write_pilot_plan,
    write_pilot_reports,
)
from analyze_frequency_response_fft import analyze_frequency_response_fft  # noqa: E402


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_map(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(candidate["candidate_id"]): candidate for candidate in plan["candidates"]}


def _prepare_assets(
    workspace: Path, output: Path, plan: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    for candidate in plan["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        generated = create_runtime_assets(
            EvaluationParameters(
                kp=float(candidate["kp"]),
                ki=float(candidate["ki"]),
                i_max=float(candidate["i_max"]),
            ),
            output / "assets" / candidate_id,
            source_urdf=workspace / "src/rbkairos/urdf/rbkairos.urdf",
            source_mjcf=workspace / "src/rbkairos/mujoco/rbkairos.xml",
            source_traction_scenes={
                axis: workspace
                / "src/rbkairos/mujoco"
                / f"rbkairos_scene_tendon_test_{axis}.xml"
                for axis in ("x", "y")
            },
            source_config=workspace / "src/rbkairos/config/mujoco_mecanum_drive.yaml",
        )
        assets[candidate_id] = generated.to_dict()
    _write_json(output / "asset_index.json", assets)
    return assets


def _launch_command(assets: Mapping[str, Any]) -> list[str]:
    return [
        "ros2",
        "launch",
        "rbkairos",
        "mujoco_mecanum_drive.launch.py",
        f"urdf_file:={assets['urdf_path']}",
        f"config_file:={assets['config_path']}",
        "model_package:=rbkairos",
        f"model_file:={assets['mujoco_model_path']}",
        "traction_tendon_enabled:=false",
    ]


def _recorder_command(run: Mapping[str, Any], raw_dir: Path) -> list[str]:
    return [
        "ros2",
        "run",
        "rbkairos",
        "frequency_response_recorder.py",
        "--environment",
        "sim",
        "--axis",
        "x",
        "--amplitude",
        "0.10",
        "--frequencies",
        str(run["frequency_hz"]),
        "--cycles",
        "6.0",
        "--discard-cycles",
        "2.0",
        "--ramp-cycles",
        "1.0",
        "--publish-rate",
        "200.0",
        "--settle-between-frequencies",
        "1.0",
        "--max-x",
        "1.5",
        "--max-y",
        "0.15",
        "--max-yaw-deg",
        "10.0",
        "--require-runtime-metadata",
        "--runtime-metadata-timeout",
        "5.0",
        "--write-velocity-pid-diagnostics",
        "--output-dir",
        str(raw_dir),
    ]


def active_controllers(output: str) -> set[str]:
    clean = ANSI_ESCAPE.sub("", output)
    return {
        fields[0]
        for line in clean.splitlines()
        if len(fields := line.split()) >= 3 and fields[-1] == "active"
    }


def _wait_for_controllers(
    process: subprocess.Popen[Any], timeout_s: float = 20.0
) -> dict[str, Any]:
    started = time.monotonic()
    last_stdout = ""
    last_stderr = ""
    while time.monotonic() - started < timeout_s:
        if process.poll() is not None:
            raise RuntimeError(f"MuJoCo launch exited early with {process.returncode}")
        completed = subprocess.run(
            ["ros2", "control", "list_controllers"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3.0,
        )
        last_stdout = completed.stdout or ""
        last_stderr = completed.stderr or ""
        active = active_controllers(last_stdout)
        required = {
            "joint_state_broadcaster",
            "mecanum_base_controller",
            "ur5_joint_trajectory_controller",
        }
        if completed.returncode == 0 and required <= active:
            return {
                "wait_s": time.monotonic() - started,
                "active": sorted(active),
                "stdout": last_stdout,
            }
        time.sleep(0.25)
    raise RuntimeError(
        "controllers did not become active; stdout=" + last_stdout + "; stderr=" + last_stderr
    )


def _terminate_group(process: subprocess.Popen[Any], timeout_s: float = 12.0) -> dict[str, Any]:
    result = {
        "pid": process.pid,
        "sigterm": False,
        "sigkill": False,
        "returncode": process.poll(),
    }
    if process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            result["sigterm"] = True
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            result["sigkill"] = True
        except ProcessLookupError:
            pass
        process.wait(timeout=3.0)
    result["returncode"] = process.returncode
    return result


def _wait_graph_cleanup(timeout_s: float = 10.0) -> dict[str, Any]:
    started = time.monotonic()
    blocking = {"/mujoco_sim_ros2_node", "/controller_manager", "/robot_state_publisher"}
    last_nodes: list[str] = []
    while time.monotonic() - started < timeout_s:
        completed = subprocess.run(
            ["ros2", "node", "list", "--no-daemon", "--spin-time", "0.1"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3.0,
        )
        last_nodes = [
            line.strip() for line in completed.stdout.splitlines() if line.strip().startswith("/")
        ]
        if completed.returncode == 0 and not (blocking & set(last_nodes)):
            return {"clean": True, "wait_s": time.monotonic() - started, "nodes": last_nodes}
        time.sleep(0.25)
    return {"clean": False, "wait_s": time.monotonic() - started, "nodes": last_nodes}


def _find_recorder_run(raw_dir: Path) -> Path:
    candidates = [
        path.parent
        for path in raw_dir.rglob("run_metadata.json")
        if path.parent.name.startswith("f_") and (path.parent / "wheel_velocity_pid.csv").is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one FRF run under {raw_dir}, got {len(candidates)}")
    return candidates[0]


def _run_one(
    output: Path,
    run: Mapping[str, Any],
    candidate: Mapping[str, Any],
    assets: Mapping[str, Any],
) -> dict[str, Any]:
    run_root = output / "candidates" / str(candidate["candidate_id"]) / "runs" / str(run["run_id"])
    raw_dir = run_root / "raw"
    logs = run_root / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    launch_command = _launch_command(assets)
    recorder_command = _recorder_command(run, raw_dir)
    execution: dict[str, Any] = {
        "run": dict(run),
        "candidate": dict(candidate),
        "launch_command": launch_command,
        "recorder_command": recorder_command,
        "started_wall_time": time.time(),
    }
    environment = os.environ.copy()
    ros_log_dir = logs / "ros"
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    environment["ROS_LOG_DIR"] = str(ros_log_dir)
    launch_stdout = (logs / "launch_stdout.log").open("w", encoding="utf-8")
    launch_stderr = (logs / "launch_stderr.log").open("w", encoding="utf-8")
    launch = subprocess.Popen(
        launch_command,
        stdout=launch_stdout,
        stderr=launch_stderr,
        env=environment,
        start_new_session=True,
    )
    try:
        execution["controller_readiness"] = _wait_for_controllers(launch)
        frequency = float(run["frequency_hz"])
        timeout_s = 6.0 / frequency + 35.0
        started = time.monotonic()
        with (logs / "recorder_stdout.log").open("w", encoding="utf-8") as stdout, (
            logs / "recorder_stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                recorder_command,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                check=False,
                timeout=timeout_s,
            )
        execution["recorder_returncode"] = completed.returncode
        execution["recorder_wall_duration_s"] = time.monotonic() - started
    finally:
        execution["launch_cleanup"] = _terminate_group(launch)
        launch_stdout.close()
        launch_stderr.close()
    execution["graph_cleanup"] = _wait_graph_cleanup()
    execution["finished_wall_time"] = time.time()
    if execution.get("recorder_returncode") != 0:
        _write_json(run_root / "execution.json", execution)
        raise RuntimeError(f"FRF recorder failed for {run['run_id']}")
    if not execution["graph_cleanup"]["clean"]:
        _write_json(run_root / "execution.json", execution)
        raise RuntimeError(f"ROS graph did not clean up after {run['run_id']}")
    recorder_run = _find_recorder_run(raw_dir)
    execution["source_run_dir"] = str(recorder_run)
    _write_json(run_root / "execution.json", execution)
    return execution


def _existing_execution(run_root: Path) -> dict[str, Any] | None:
    path = run_root / "execution.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("recorder_returncode") != 0:
        return None
    source_run = Path(str(value.get("source_run_dir", "")))
    if not (source_run / "wheel_velocity_pid.csv").is_file():
        return None
    return value


def _analyze(
    workspace: Path,
    output: Path,
    plan: Mapping[str, Any],
    real_summary: Path,
) -> dict[str, Any]:
    real_gains = real_gain_map(real_summary)
    rows: list[dict[str, Any]] = []
    candidate_by_id = _candidate_map(plan)
    for candidate_id, candidate in candidate_by_id.items():
        candidate_root = output / "candidates" / candidate_id
        analysis_dir = candidate_root / "fft_analysis"
        analyze_frequency_response_fft(
            candidate_root,
            output_dir=analysis_dir,
            environment="mujoco",
            output_source="odom_twist_vx",
            discard_cycles=2.0,
            ramp_cycles=1.0,
            analysis_cycles=4,
            min_analysis_cycles=2.0,
            grid_rate_hz=100.0,
            max_harmonic=7,
            nyquist_safety=0.90,
            plots=True,
            html=True,
        )
        pid_paths: dict[float, Path] = {}
        for run in plan["runs"]:
            if run["candidate_id"] != candidate_id:
                continue
            run_root = candidate_root / "runs" / str(run["run_id"])
            execution = _existing_execution(run_root)
            if execution is None:
                raise PilotInputError(f"missing completed execution for {run['run_id']}")
            pid_paths[float(run["frequency_hz"])] = (
                Path(str(execution["source_run_dir"])) / "wheel_velocity_pid.csv"
            )
        rows.extend(
            build_candidate_rows(
                candidate,
                analysis_dir / "frequency_fft_summary.csv",
                pid_paths,
                real_gains,
            )
        )
    report = write_pilot_reports(rows, output)
    return {"report": report, "rows": rows}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--real-summary",
        type=Path,
        default=Path(
            "data_real_lab/20260625_142549_real_lab_x_A0p10/analysis/frequency_summary.csv"
        ),
    )
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = default_pilot_plan()
    write_pilot_plan(plan, output)
    assets = _prepare_assets(workspace, output, plan)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "run_count": len(plan["runs"])}))
        return 0

    candidates = _candidate_map(plan)
    selected = list(plan["runs"])
    if args.run_id:
        requested = set(args.run_id)
        selected = [run for run in selected if run["run_id"] in requested]
        missing = requested - {str(run["run_id"]) for run in selected}
        if missing:
            raise PilotInputError("unknown run ids: " + ", ".join(sorted(missing)))

    if args.execute:
        for index, run in enumerate(selected, start=1):
            candidate = candidates[str(run["candidate_id"])]
            run_root = (
                output
                / "candidates"
                / str(candidate["candidate_id"])
                / "runs"
                / str(run["run_id"])
            )
            existing = None if args.no_resume else _existing_execution(run_root)
            if existing is not None:
                print(f"[{index}/{len(selected)}] resume {run['run_id']}", flush=True)
                continue
            print(f"[{index}/{len(selected)}] run {run['run_id']}", flush=True)
            _run_one(output, run, candidate, assets[str(candidate["candidate_id"])])

    if args.run_id:
        print(json.dumps({"status": "partial_complete", "run_count": len(selected)}))
        return 0

    result = _analyze(workspace, output, plan, (workspace / args.real_summary).resolve())
    print(
        json.dumps(
            {
                "status": "complete",
                "run_count": len(plan["runs"]),
                "report": str(output / "frf_i_max_sensitivity_report.json"),
                "global": result["report"]["global"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
