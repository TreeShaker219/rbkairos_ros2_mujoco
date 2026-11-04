#!/usr/bin/env python3
"""Prepare, execute, resume, and analyze the isolated D2 i_max pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Mapping, Sequence
import re

from traction_tendon_i_max_pilot import (
    PilotInputError,
    analyze_run_directory,
    default_pilot_plan,
    prepare_candidate_assets,
    write_matrix_reports,
    write_pilot_plan,
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_map(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item["candidate_id"]): item for item in plan["candidates"]}


def _prepare(workspace: Path, output: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    manifests: dict[str, Any] = {}
    for candidate in plan["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        manifests[candidate_id] = prepare_candidate_assets(
            candidate,
            output / "assets" / candidate_id,
            source_urdf=workspace / "src/rbkairos/urdf/rbkairos.urdf",
            source_mjcf=workspace / "src/rbkairos/mujoco/rbkairos.xml",
            source_scenes={
                axis: workspace / "src/rbkairos/mujoco" / f"rbkairos_scene_tendon_test_{axis}.xml"
                for axis in ("x", "y")
            },
        )
    _write_json(output / "asset_index.json", manifests)
    return manifests


def _axis_names(axis: str) -> dict[str, str]:
    return {
        "tendon": f"traction_cable_{axis}",
        "sensor": f"traction_cable_{axis}_limit_force",
        "attach": f"traction_attach_{axis}",
        "anchor": f"traction_anchor_{axis}",
    }


def _launch_command(run: Mapping[str, Any], assets: Mapping[str, Any]) -> list[str]:
    axis = str(run["axis"])
    names = _axis_names(axis)
    return [
        "ros2", "launch", "rbkairos", "mujoco_mecanum_drive.launch.py",
        f"urdf_file:={assets['urdf_path']}",
        f"model_file:={assets['scene_paths'][axis]}",
        "model_package:=rbkairos",
        "traction_tendon_enabled:=true",
        f"traction_tendon_name:={names['tendon']}",
        f"traction_tendon_sensor_name:={names['sensor']}",
        f"traction_tendon_attach_site_name:={names['attach']}",
        f"traction_tendon_anchor_site_name:={names['anchor']}",
        "traction_tendon_publish_rate_hz:=100.0",
        "traction_tendon_state_topic:=/traction_tendon/state",
        "traction_tendon_wrench_topic:=/traction_tendon/wrench",
        "traction_tendon_frame_id:=world",
    ]


def _recorder_command(
    run: Mapping[str, Any], assets: Mapping[str, Any], raw_dir: Path
) -> list[str]:
    axis = str(run["axis"])
    names = _axis_names(axis)
    velocity_flag = "--velocity-x" if axis == "x" else "--velocity-y"
    return [
        "ros2", "run", "rbkairos", "traction_tendon_experiment.py",
        "--axis", axis,
        velocity_flag, str(run["speed_mps"]),
        "--warmup-duration", "1.0",
        "--ramp-duration", "0.5",
        "--min-observation-duration", "2.0",
        "--max-wait-duration", str(run["constant_duration_s"]),
        "--post-stop-duration", "0.5",
        "--steady-window", "2.0",
        "--steady-window-step", "0.25",
        "--required-stable-windows", "4",
        "--publish-rate", "100.0",
        "--observer-publish-rate", "100.0",
        "--startup-timeout", "15.0",
        "--topic-timeout", "1.0",
        "--min-active-ratio", "0.95",
        "--max-force-cv", "0.05",
        "--max-force-slope", "0.5",
        "--max-constraint-extension", "0.005",
        "--max-steady-base-velocity", "0.005",
        "--max-i-term-slope", "0.005",
        "--min-i-clamp-ratio", "0.95",
        "--abort-extension", "0.02",
        "--abort-tension", "2000.0",
        "--max-x", "0.25",
        "--max-y", "0.25",
        "--max-yaw-deg", "10.0",
        "--replicate", str(run["replicate"]),
        "--solver-variant", "i_max_pilot",
        "--run-label", str(run["run_id"]),
        "--output-dir", str(raw_dir),
        "--model-file", str(assets["scene_paths"][axis]),
        "--urdf-file", str(assets["urdf_path"]),
        "--tendon-name", names["tendon"],
        "--sensor-name", names["sensor"],
        "--attach-site-name", names["attach"],
        "--anchor-site-name", names["anchor"],
        "--expected-ur5-preset", "up",
        "--ur5-position-tolerance", "0.01",
        "--ur5-velocity-tolerance", "0.01",
    ]


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


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
    result = {"pid": process.pid, "sigterm": False, "sigkill": False, "returncode": process.poll()}
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
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip().startswith("/")
        ]
        if completed.returncode == 0 and not (blocking & set(last_nodes)):
            return {"clean": True, "wait_s": time.monotonic() - started, "nodes": last_nodes}
        time.sleep(0.25)
    return {"clean": False, "wait_s": time.monotonic() - started, "nodes": last_nodes}


def _find_recorder_run(raw_dir: Path) -> Path:
    candidates = [path.parent for path in raw_dir.glob("*/run_metadata.json")]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one recorder directory under {raw_dir}, "
            f"got {len(candidates)}"
        )
    return candidates[0]


def _run_one(
    output: Path,
    run: Mapping[str, Any],
    candidate: Mapping[str, Any],
    assets: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    run_root = output / "runs" / str(run["run_id"])
    raw_dir = run_root / "raw"
    logs = run_root / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "pilot_run_spec.json", {"run": dict(run), "candidate": dict(candidate)})
    launch_command = _launch_command(run, assets)
    recorder_command = _recorder_command(run, assets, raw_dir)
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
                timeout=float(run["constant_duration_s"]) + 35.0,
            )
        execution["recorder_returncode"] = completed.returncode
        execution["recorder_wall_duration_s"] = time.monotonic() - started
    finally:
        execution["launch_cleanup"] = _terminate_group(launch)
        launch_stdout.close()
        launch_stderr.close()
    execution["graph_cleanup"] = _wait_graph_cleanup()
    execution["finished_wall_time"] = time.time()
    _write_json(run_root / "execution.json", execution)
    if not execution["graph_cleanup"]["clean"]:
        raise RuntimeError("ROS graph did not clean up after " + str(run["run_id"]))
    recorder_run = _find_recorder_run(raw_dir)
    summary = analyze_run_directory(recorder_run, run, candidate, write_outputs=True)
    _write_json(run_root / "pilot_summary.json", summary)
    return recorder_run, summary


def _load_existing_summary(run_root: Path) -> dict[str, Any] | None:
    path = run_root / "pilot_summary.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _analyze_existing(
    output: Path,
    plan: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for run in plan["runs"]:
        run_root = output / "runs" / str(run["run_id"])
        raw_dir = run_root / "raw"
        recorder_run = _find_recorder_run(raw_dir)
        summary = analyze_run_directory(
            recorder_run,
            run,
            candidates[str(run["candidate_id"])],
            write_outputs=True,
        )
        _write_json(run_root / "pilot_summary.json", summary)
        summaries.append(summary)
    return summaries


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-id", action="append", default=[], help="optional exact run selection"
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = default_pilot_plan()
    write_pilot_plan(plan, output)
    assets = _prepare(workspace, output, plan)
    if args.prepare_only:
        print(json.dumps({
            "status": "prepared",
            "output_dir": str(output),
            "run_count": len(plan["runs"]),
        }, sort_keys=True))
        return 0

    candidates = _candidate_map(plan)
    selected = list(plan["runs"])
    if args.run_id:
        requested = set(args.run_id)
        selected = [run for run in selected if run["run_id"] in requested]
        missing = requested - {run["run_id"] for run in selected}
        if missing:
            raise PilotInputError("unknown run ids: " + ", ".join(sorted(missing)))
    if args.analyze_only:
        if args.run_id:
            raise PilotInputError("--run-id is not supported with --analyze-only")
        summaries = _analyze_existing(output, plan, candidates)
    else:
        summaries = []
        for index, run in enumerate(selected, start=1):
            run_root = output / "runs" / str(run["run_id"])
            existing = None if args.no_resume else _load_existing_summary(run_root)
            if existing is not None:
                summaries.append(existing)
                print(f"[{index}/{len(selected)}] resume {run['run_id']}", flush=True)
                continue
            print(f"[{index}/{len(selected)}] run {run['run_id']}", flush=True)
            _, summary = _run_one(
                output,
                run,
                candidates[str(run["candidate_id"])],
                assets[str(run["candidate_id"])],
            )
            summaries.append(summary)
        if len(selected) != len(plan["runs"]):
            _write_json(output / "partial_selection_summary.json", {"runs": summaries})
            print(json.dumps({
                "status": "partial_complete",
                "run_count": len(summaries),
            }, sort_keys=True))
            return 0
        # A resumed full run may have summaries loaded in matrix order already.
    reports = write_matrix_reports(summaries, output)
    aggregate = reports["aggregate"]
    print(json.dumps({
        "status": "complete",
        "run_count": len(summaries),
        "i_max_identifiable": aggregate["i_max_identifiable"],
        "report": reports["report_json"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
