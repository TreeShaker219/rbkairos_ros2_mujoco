from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExperimentRunRef:
    experiment: str
    command: list[str]
    output_dir: Path
    returncode: int | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    timed_out: bool = False
    timeout_s: float | None = None
    duration_s: float | None = None
    discovered_output: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "command": self.command,
            "output_dir": str(self.output_dir),
            "returncode": self.returncode,
            "stdout_path": str(self.stdout_path) if self.stdout_path else None,
            "stderr_path": str(self.stderr_path) if self.stderr_path else None,
            "timed_out": self.timed_out,
            "timeout_s": self.timeout_s,
            "duration_s": self.duration_s,
            "discovered_output": (
                str(self.discovered_output) if self.discovered_output else None
            ),
        }


def run_command_ref(
    experiment: str,
    command: list[str],
    output_dir: Path,
    log_dir: Path,
    execute: bool = False,
    timeout_s: float | None = None,
) -> ExperimentRunRef:
    log_dir.mkdir(parents=True, exist_ok=True)
    ref = ExperimentRunRef(
        experiment=experiment,
        command=command,
        output_dir=output_dir,
        stdout_path=log_dir / f"{experiment}_stdout.log",
        stderr_path=log_dir / f"{experiment}_stderr.log",
        timeout_s=timeout_s,
    )
    if not execute:
        return ref

    started = time.monotonic()
    with ref.stdout_path.open("w", encoding="utf-8") as stdout, ref.stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        try:
            completed = subprocess.run(
                command,
                stdout=stdout,
                stderr=stderr,
                check=False,
                timeout=timeout_s,
            )
            ref.returncode = completed.returncode
        except subprocess.TimeoutExpired:
            ref.timed_out = True
            ref.returncode = None
    ref.duration_s = time.monotonic() - started
    return ref


def build_traction_command(args: list[str]) -> list[str]:
    return ["ros2", "run", "rbkairos", "traction_tendon_experiment.py", *args]


def build_frf_command(args: list[str]) -> list[str]:
    return ["ros2", "run", "rbkairos", "frequency_response_recorder.py", *args]


def _append_arg(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    if isinstance(value, (list, tuple)):
        command.append(flag)
        command.extend(str(item) for item in value)
        return
    command.extend([flag, str(value)])


def build_traction_command_from_config(config: dict[str, Any], output_dir: Path) -> list[str]:
    args: list[str] = []
    for key, flag in [
        ("axis", "--axis"),
        ("velocity_x", "--velocity-x"),
        ("velocity_y", "--velocity-y"),
        ("warmup_duration", "--warmup-duration"),
        ("ramp_duration", "--ramp-duration"),
        ("min_observation_duration", "--min-observation-duration"),
        ("max_wait_duration", "--max-wait-duration"),
        ("post_stop_duration", "--post-stop-duration"),
        ("steady_window", "--steady-window"),
        ("steady_window_step", "--steady-window-step"),
        ("required_stable_windows", "--required-stable-windows"),
        ("publish_rate", "--publish-rate"),
        ("observer_publish_rate", "--observer-publish-rate"),
        ("startup_timeout", "--startup-timeout"),
        ("topic_timeout", "--topic-timeout"),
        ("max_warmup_tension", "--max-warmup-tension"),
        ("max_warmup_active_ratio", "--max-warmup-active-ratio"),
        ("min_active_ratio", "--min-active-ratio"),
        ("max_force_cv", "--max-force-cv"),
        ("max_force_slope", "--max-force-slope"),
        ("max_constraint_extension", "--max-constraint-extension"),
        ("max_steady_base_velocity", "--max-steady-base-velocity"),
        ("max_i_term_slope", "--max-i-term-slope"),
        ("min_i_clamp_ratio", "--min-i-clamp-ratio"),
        ("abort_extension", "--abort-extension"),
        ("abort_tension", "--abort-tension"),
        ("max_x", "--max-x"),
        ("max_y", "--max-y"),
        ("max_yaw_deg", "--max-yaw-deg"),
        ("replicate", "--replicate"),
        ("solver_variant", "--solver-variant"),
        ("run_label", "--run-label"),
        ("model_file", "--model-file"),
        ("urdf_file", "--urdf-file"),
        ("tendon_name", "--tendon-name"),
        ("sensor_name", "--sensor-name"),
        ("attach_site_name", "--attach-site-name"),
        ("anchor_site_name", "--anchor-site-name"),
        ("expected_ur5_preset", "--expected-ur5-preset"),
        ("ur5_position_tolerance", "--ur5-position-tolerance"),
        ("ur5_velocity_tolerance", "--ur5-velocity-tolerance"),
    ]:
        _append_arg(args, flag, config.get(key))
    args.extend(["--output-dir", str(output_dir)])
    return build_traction_command(args)


def build_frf_command_from_config(config: dict[str, Any], output_dir: Path) -> list[str]:
    args: list[str] = []
    for key, flag in [
        ("environment", "--environment"),
        ("axis", "--axis"),
        ("amplitude", "--amplitude"),
        ("frequencies", "--frequencies"),
        ("publish_rate", "--publish-rate"),
        ("cycles", "--cycles"),
        ("discard_cycles", "--discard-cycles"),
        ("ramp_cycles", "--ramp-cycles"),
        ("max_x", "--max-x"),
        ("max_y", "--max-y"),
        ("max_yaw_deg", "--max-yaw-deg"),
        ("odom_timeout", "--odom-timeout"),
        ("runtime_metadata_timeout", "--runtime-metadata-timeout"),
        ("settle_between_frequencies", "--settle-between-frequencies"),
    ]:
        _append_arg(args, flag, config.get(key))
    _append_arg(
        args,
        "--require-runtime-metadata",
        bool(config.get("require_runtime_metadata", False)),
    )
    for key, flag in [
        ("write_full_dof_state", "--write-full-dof-state"),
        ("write_raw_dof_state", "--write-raw-dof-state"),
        ("write_contact_forces", "--write-contact-forces"),
        ("require_pid_nonbinding_evidence", "--require-pid-nonbinding-evidence"),
    ]:
        _append_arg(args, flag, bool(config.get(key, False)))
    args.extend(["--output-dir", str(output_dir)])
    return build_frf_command(args)


def latest_traction_tendon_run(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    candidates: list[Path] = []
    for path in base_dir.iterdir():
        metadata_path = path / "run_metadata.json"
        if not path.is_dir() or not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(metadata, dict)
            and metadata.get("experiment_type") == "traction_tendon_pull"
            and metadata.get("measurement_method") == "tendon_limit"
            and metadata.get("schema_version") == 1
            and (path / "traction_tendon_state.csv").is_file()
        ):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def latest_frequency_response_experiment(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    candidates = [
        path
        for path in base_dir.iterdir()
        if path.is_dir() and (path / "metadata.json").exists() and (path / "runs").is_dir()
    ]
    if not candidates:
        candidates = [
            path for path in base_dir.iterdir() if path.is_dir() and (path / "runs").is_dir()
        ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def traction_timeout_from_config(config: dict[str, Any], margin_s: float) -> float:
    return (
        float(config.get("startup_timeout", 10.0))
        + float(config.get("warmup_duration", 1.0))
        + float(config.get("ramp_duration", 0.5))
        + float(config.get("max_wait_duration", 60.0))
        + float(config.get("post_stop_duration", 0.5))
        + float(margin_s)
    )


def frf_timeout_from_config(config: dict[str, Any], margin_s: float) -> float:
    frequencies = [float(value) for value in config.get("frequencies", [])]
    if not frequencies:
        frequencies = [0.05, 0.10, 0.20, 0.50, 0.80, 1.20, 2.00]
    cycles = float(config.get("cycles", 6.0))
    settle = float(config.get("settle_between_frequencies", 1.0))
    planned = sum(cycles / frequency for frequency in frequencies)
    return planned + settle * len(frequencies) + float(margin_s)
