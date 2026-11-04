#!/usr/bin/env python3
"""
Analyze RB-KAIROS MuJoCo base velocity frequency-response recordings.

Analyze one recorder experiment directory:
  ros2 run rbkairos analyze_frequency_response.py \
    --experiment-dir data/frequency_response/20260604_153000_sim_x_A0p05

Analyze a grouped directory containing multiple single-frequency recorder outputs:
  ros2 run rbkairos analyze_frequency_response.py \
    --experiment-dir data/frequency_response/202606051545

The script also supports direct path execution:
  python3 src/rbkairos/scripts/analyze_frequency_response.py --experiment-dir <dir>
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import site
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np


DEFAULT_WHEEL_DOF_ORDER = (
    "left_front_wheel_joint",
    "right_front_wheel_joint",
    "left_back_wheel_joint",
    "right_back_wheel_joint",
)
DEFAULT_WHEEL_DOF_NAMES = set(DEFAULT_WHEEL_DOF_ORDER)
WHEEL_QFRC_SUMMARY_COLUMNS = {
    name: f"{name}_qfrc_applied" for name in DEFAULT_WHEEL_DOF_ORDER
}
DEFAULT_WHEEL_EFFORT_LIMIT_NM = 100.0
SATURATION_FRACTION = 0.98

SUMMARY_FIELDS = [
    "run_id",
    "source_experiment",
    "source_experiment_dir",
    "source_run_dir",
    "environment",
    "axis",
    "f_hz",
    "omega_rad_s",
    "A_cmd_nominal",
    "A_in_fit",
    "A_out_fit",
    "gain_linear",
    "gain_db",
    "phase_rad_optional",
    "cmd_fit_rmse",
    "cmd_fit_r2",
    "out_fit_rmse",
    "out_fit_r2",
    "offset",
    "drift_rate",
    "final_dx",
    "final_dy",
    "final_dyaw",
    "max_abs_x",
    "max_abs_y",
    "max_abs_yaw",
    "saturation_ratio",
    "contact_force_rms",
    "output_source",
    "valid",
    "reject_reason",
]
SUMMARY_NUMERIC_FIELDS = {
    "f_hz",
    "omega_rad_s",
    "A_cmd_nominal",
    "A_in_fit",
    "A_out_fit",
    "gain_linear",
    "gain_db",
    "phase_rad_optional",
    "cmd_fit_rmse",
    "cmd_fit_r2",
    "out_fit_rmse",
    "out_fit_r2",
    "offset",
    "drift_rate",
    "final_dx",
    "final_dy",
    "final_dyaw",
    "max_abs_x",
    "max_abs_y",
    "max_abs_yaw",
    "saturation_ratio",
    "contact_force_rms",
}


@dataclass
class FitResult:
    amplitude: float
    phase_rad: float
    offset: float
    drift_rate: float
    rmse: float
    r2: float
    fitted: np.ndarray


@dataclass
class RunAnalysisData:
    run_id: str
    plot_id: str
    frequency_hz: float
    omega_rad_s: float
    t: np.ndarray
    cmd: np.ndarray
    output: np.ndarray
    cmd_fit: np.ndarray
    output_fit: np.ndarray
    output_source: str


@dataclass(frozen=True)
class FrequencyAnalysisOptions:
    resample_rate: float = 100.0
    default_discard_cycles: float = 2.0


def float_or_nan(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        text = str(value).strip()
        if text == "":
            return math.nan
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def bool_from_metadata(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no"}
    return bool(value)


def wheel_effort_limits_from_metadata(metadata: dict[str, Any]) -> dict[str, float]:
    """Return per-wheel effort limits, with explicit defaults for legacy recordings."""
    controller_facts = metadata.get("controller_facts")
    if not isinstance(controller_facts, dict):
        controller_facts = {}

    raw_names = controller_facts.get("wheel_dof_names")
    if isinstance(raw_names, list) and all(isinstance(name, str) for name in raw_names):
        wheel_names = set(raw_names)
    else:
        wheel_names = set(DEFAULT_WHEEL_DOF_NAMES)

    common_pid = controller_facts.get("wheel_velocity_pid")
    if not isinstance(common_pid, dict):
        common_pid = {}
    common_limit = float_or_nan(common_pid.get("effort_limit_nm"))

    by_joint = controller_facts.get("wheel_velocity_pid_by_joint")
    if not isinstance(by_joint, dict):
        by_joint = {}

    limits: dict[str, float] = {}
    for wheel_name in wheel_names:
        joint_pid = by_joint.get(wheel_name)
        if not isinstance(joint_pid, dict):
            joint_pid = {}
        effort_limit = float_or_nan(joint_pid.get("effort_limit_nm"))
        if not math.isfinite(effort_limit) or effort_limit <= 0.0:
            effort_limit = common_limit
        if not math.isfinite(effort_limit) or effort_limit <= 0.0:
            effort_limit = DEFAULT_WHEEL_EFFORT_LIMIT_NM
        limits[wheel_name] = effort_limit
    return limits


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def normalize_frequency_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one frequency_summary.csv row into typed analyzer fields."""
    normalized = dict(row)
    for field in SUMMARY_NUMERIC_FIELDS:
        normalized[field] = float_or_nan(row.get(field))
    normalized["valid"] = bool_from_metadata(row.get("valid"), default=False)
    normalized["reject_reason"] = str(row.get("reject_reason") or "")
    return normalized


def read_frequency_summary(path: Path) -> list[dict[str, Any]]:
    """Read frequency_summary.csv using the analyzer's field semantics."""
    return [normalize_frequency_summary_row(row) for row in read_csv_rows(path)]


def valid_frequency_summary_rows(rows: list[dict[str, Any]]) -> dict[float, dict[str, Any]]:
    """Return valid summary rows keyed by rounded frequency in Hz."""
    valid: dict[float, dict[str, Any]] = {}
    for row in rows:
        normalized = normalize_frequency_summary_row(row)
        frequency = float_or_nan(normalized.get("f_hz"))
        gain_db = float_or_nan(normalized.get("gain_db"))
        if (
            math.isfinite(frequency)
            and math.isfinite(gain_db)
            and bool_from_metadata(normalized.get("valid"), default=False)
        ):
            valid[round(frequency, 9)] = normalized
    return valid


def column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.array([float_or_nan(row.get(name)) for row in rows], dtype=float)


def finite_min(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else math.nan


def finite_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else math.nan


def run_sort_key(run_dir: Path) -> tuple[float, str]:
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        try:
            metadata = read_json(metadata_path)
            return (float(metadata.get("frequency_hz", math.inf)), run_dir.name)
        except Exception:
            pass
    text = run_dir.name.removeprefix("f_").removesuffix("Hz").replace("p", ".")
    return (float_or_nan(text), run_dir.name)


def sanitize_token(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return sanitized.strip("._-") or "unnamed"


def fit_sine(t: np.ndarray, y: np.ndarray, omega: float) -> Optional[FitResult]:
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if t.size < 4:
        return None

    t_drift = t - t[0]
    design = np.column_stack(
        [
            np.ones_like(t),
            t_drift,
            np.sin(omega * t),
            np.cos(omega * t),
        ]
    )
    try:
        coeff, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    fitted = design @ coeff
    residual = y - fitted
    rmse = float(np.sqrt(np.mean(residual * residual)))
    ss_res = float(np.sum(residual * residual))
    centered = y - np.mean(y)
    ss_tot = float(np.sum(centered * centered))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else math.nan
    a = float(coeff[2])
    b = float(coeff[3])
    return FitResult(
        amplitude=math.hypot(a, b),
        phase_rad=math.atan2(b, a),
        offset=float(coeff[0]),
        drift_rate=float(coeff[1]),
        rmse=rmse,
        r2=r2,
        fitted=fitted,
    )


def resample_to_grid(
    source_t: np.ndarray,
    source_y: np.ndarray,
    grid_t: np.ndarray,
) -> Optional[np.ndarray]:
    mask = np.isfinite(source_t) & np.isfinite(source_y)
    if np.count_nonzero(mask) < 2:
        return None
    t = source_t[mask]
    y = source_y[mask]
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    unique_t, unique_indices = np.unique(t, return_index=True)
    unique_y = y[unique_indices]
    if unique_t.size < 2:
        return None
    return np.interp(grid_t, unique_t, unique_y)


def steady_grid(
    cmd_t: np.ndarray,
    out_t: np.ndarray,
    discard_time: float,
    resample_rate: float,
) -> Optional[np.ndarray]:
    cmd_min = finite_min(cmd_t)
    cmd_max = finite_max(cmd_t)
    out_min = finite_min(out_t)
    out_max = finite_max(out_t)
    if not all(math.isfinite(value) for value in (cmd_min, cmd_max, out_min, out_max)):
        return None

    start = max(discard_time, cmd_min, out_min)
    stop = min(cmd_max, out_max)
    dt = 1.0 / resample_rate
    if stop <= start + dt:
        return None
    return np.arange(start, stop + 0.5 * dt, dt, dtype=float)


def compute_motion_summary(odom_rows: list[dict[str, str]], metadata: dict[str, Any]) -> dict[str, float]:
    x = column(odom_rows, "x")
    y = column(odom_rows, "y")
    yaw = column(odom_rows, "yaw")
    x0 = float_or_nan(metadata.get("x0"))
    y0 = float_or_nan(metadata.get("y0"))
    yaw0 = float_or_nan(metadata.get("yaw0"))

    if not math.isfinite(x0) and x.size:
        x0 = float(x[0])
    if not math.isfinite(y0) and y.size:
        y0 = float(y[0])
    if not math.isfinite(yaw0) and yaw.size:
        yaw0 = float(yaw[0])

    dx = x - x0 if math.isfinite(x0) else np.full_like(x, math.nan)
    dy = y - y0 if math.isfinite(y0) else np.full_like(y, math.nan)
    dyaw = np.array([wrap_angle(value - yaw0) for value in yaw]) if math.isfinite(yaw0) else np.full_like(yaw, math.nan)

    return {
        "final_dx": float(dx[np.isfinite(dx)][-1]) if np.isfinite(dx).any() else math.nan,
        "final_dy": float(dy[np.isfinite(dy)][-1]) if np.isfinite(dy).any() else math.nan,
        "final_dyaw": float(dyaw[np.isfinite(dyaw)][-1]) if np.isfinite(dyaw).any() else math.nan,
        "max_abs_x": float(np.max(np.abs(dx[np.isfinite(dx)]))) if np.isfinite(dx).any() else math.nan,
        "max_abs_y": float(np.max(np.abs(dy[np.isfinite(dy)]))) if np.isfinite(dy).any() else math.nan,
        "max_abs_yaw": float(np.max(np.abs(dyaw[np.isfinite(dyaw)]))) if np.isfinite(dyaw).any() else math.nan,
    }


def compute_saturation_ratio_from_wheel_summary(
    path: Path,
    discard_time: float,
    wheel_effort_limits: dict[str, float],
) -> float:
    rows = read_csv_rows(path)
    total = 0
    saturated = 0
    for row in rows:
        t = float_or_nan(row.get("t_experiment"))
        if math.isfinite(t) and t < discard_time:
            continue
        for dof_name, effort_limit in wheel_effort_limits.items():
            applied = float_or_nan(row.get(WHEEL_QFRC_SUMMARY_COLUMNS.get(dof_name, "")))
            if not math.isfinite(applied):
                continue
            total += 1
            if abs(applied) >= SATURATION_FRACTION * effort_limit:
                saturated += 1
    return float(saturated / total) if total > 0 else math.nan


def compute_saturation_ratio_from_dof_state(
    path: Path,
    discard_time: float,
    wheel_effort_limits: dict[str, float],
) -> float:
    rows = read_csv_rows(path)
    total = 0
    saturated = 0
    for row in rows:
        t = float_or_nan(row.get("t_experiment"))
        if math.isfinite(t) and t < discard_time:
            continue
        dof_name = row.get("dof_name")
        if dof_name not in wheel_effort_limits:
            continue
        applied = float_or_nan(row.get("qfrc_applied"))
        if not math.isfinite(applied):
            continue
        total += 1
        if abs(applied) >= SATURATION_FRACTION * wheel_effort_limits[dof_name]:
            saturated += 1
    return float(saturated / total) if total > 0 else math.nan


def compute_saturation_ratio(
    run_dir: Path,
    discard_time: float,
    wheel_effort_limits: dict[str, float],
) -> float:
    summary_value = compute_saturation_ratio_from_wheel_summary(
        run_dir / "wheel_qfrc_summary.csv",
        discard_time,
        wheel_effort_limits,
    )
    if math.isfinite(summary_value):
        return summary_value
    return compute_saturation_ratio_from_dof_state(
        run_dir / "mujoco_dof_state.csv",
        discard_time,
        wheel_effort_limits,
    )


def compute_contact_force_rms_from_summary(path: Path, discard_time: float) -> float:
    rows = read_csv_rows(path)
    total_sum_force_norm_sq = 0.0
    total_count = 0
    for row in rows:
        t = float_or_nan(row.get("t_experiment"))
        if math.isfinite(t) and t < discard_time:
            continue
        count = float_or_nan(row.get("contact_count"))
        sum_force_norm_sq = float_or_nan(row.get("sum_force_norm_sq"))
        if not math.isfinite(count) or not math.isfinite(sum_force_norm_sq):
            continue
        if count <= 0.0:
            continue
        total_count += int(count)
        total_sum_force_norm_sq += sum_force_norm_sq
    if total_count <= 0:
        return math.nan
    return float(math.sqrt(total_sum_force_norm_sq / total_count))


def compute_contact_force_rms_from_contacts(path: Path, discard_time: float) -> float:
    rows = read_csv_rows(path)
    squared_norms: list[float] = []
    for row in rows:
        t = float_or_nan(row.get("t_experiment"))
        if math.isfinite(t) and t < discard_time:
            continue
        fx = float_or_nan(row.get("force_local_0"))
        fy = float_or_nan(row.get("force_local_1"))
        fz = float_or_nan(row.get("force_local_2"))
        if not all(math.isfinite(value) for value in (fx, fy, fz)):
            continue
        squared_norms.append(fx * fx + fy * fy + fz * fz)
    if not squared_norms:
        return math.nan
    return float(math.sqrt(sum(squared_norms) / len(squared_norms)))


def compute_contact_force_rms(run_dir: Path, discard_time: float) -> float:
    summary_value = compute_contact_force_rms_from_summary(
        run_dir / "contact_force_summary.csv",
        discard_time,
    )
    if math.isfinite(summary_value):
        return summary_value
    return compute_contact_force_rms_from_contacts(run_dir / "contact_forces.csv", discard_time)


def frequency_analysis_options_from_args(args: argparse.Namespace) -> FrequencyAnalysisOptions:
    return FrequencyAnalysisOptions(
        resample_rate=float(args.resample_rate),
        default_discard_cycles=float(args.default_discard_cycles),
    )


def coerce_frequency_analysis_options(options: Any) -> FrequencyAnalysisOptions:
    if isinstance(options, FrequencyAnalysisOptions):
        return options
    return frequency_analysis_options_from_args(options)


def analyze_run(
    run_dir: Path,
    source_experiment_dir: Path,
    plot_id: str,
    options: FrequencyAnalysisOptions | argparse.Namespace,
) -> tuple[dict[str, Any], Optional[RunAnalysisData]]:
    options = coerce_frequency_analysis_options(options)
    metadata_path = run_dir / "run_metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    run_id = str(metadata.get("run_id") or run_dir.name)
    frequency_hz = float_or_nan(metadata.get("frequency_hz"))
    omega = float_or_nan(metadata.get("omega_rad_s"))
    axis = str(metadata.get("axis") or "x")
    environment = str(metadata.get("environment") or "")
    amplitude_nominal = float_or_nan(metadata.get("amplitude"))
    discard_cycles = float_or_nan(metadata.get("discard_cycles"))
    if not math.isfinite(discard_cycles):
        discard_cycles = options.default_discard_cycles
    discard_time = discard_cycles / frequency_hz if frequency_hz > 0.0 else math.nan
    wheel_effort_limits = wheel_effort_limits_from_metadata(metadata)

    odom_rows = read_csv_rows(run_dir / "odom.csv")
    motion = compute_motion_summary(odom_rows, metadata)
    summary = {
        "run_id": run_id,
        "source_experiment": source_experiment_dir.name,
        "source_experiment_dir": str(source_experiment_dir),
        "source_run_dir": str(run_dir),
        "environment": environment,
        "axis": axis,
        "f_hz": frequency_hz,
        "omega_rad_s": omega,
        "A_cmd_nominal": amplitude_nominal,
        "A_in_fit": math.nan,
        "A_out_fit": math.nan,
        "gain_linear": math.nan,
        "gain_db": math.nan,
        "phase_rad_optional": math.nan,
        "cmd_fit_rmse": math.nan,
        "cmd_fit_r2": math.nan,
        "out_fit_rmse": math.nan,
        "out_fit_r2": math.nan,
        "offset": math.nan,
        "drift_rate": math.nan,
        "final_dx": motion["final_dx"],
        "final_dy": motion["final_dy"],
        "final_dyaw": motion["final_dyaw"],
        "max_abs_x": motion["max_abs_x"],
        "max_abs_y": motion["max_abs_y"],
        "max_abs_yaw": motion["max_abs_yaw"],
        "saturation_ratio": compute_saturation_ratio(
            run_dir,
            discard_time,
            wheel_effort_limits,
        ),
        "contact_force_rms": compute_contact_force_rms(run_dir, discard_time),
        "output_source": "odom_twist_vx",
        "valid": False,
        "reject_reason": "",
    }

    if not bool_from_metadata(metadata.get("valid"), default=True):
        summary["reject_reason"] = str(metadata.get("reject_reason") or "run_marked_invalid")
        return summary, None
    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0 or not math.isfinite(omega):
        summary["reject_reason"] = "invalid_frequency_metadata"
        return summary, None

    cmd_rows = read_csv_rows(run_dir / "cmd_vel.csv")
    if not cmd_rows or not odom_rows:
        summary["reject_reason"] = "missing_cmd_or_odom_samples"
        return summary, None

    cmd_t = column(cmd_rows, "t_experiment")
    cmd_name = "cmd_vx" if axis == "x" else f"cmd_v{axis}"
    cmd_y = column(cmd_rows, cmd_name)
    odom_t = column(odom_rows, "t_experiment")
    output_source = "odom_twist_vx"
    output_y = column(odom_rows, "vx")

    if not np.isfinite(output_y).any():
        output_source = "odom_position_x"
        output_y = column(odom_rows, "x")
    summary["output_source"] = output_source

    grid = steady_grid(cmd_t, odom_t, discard_time, options.resample_rate)
    if grid is None:
        summary["reject_reason"] = "insufficient_time_overlap"
        return summary, None

    steady_duration = float(grid[-1] - grid[0]) if grid.size else 0.0
    if steady_duration < 2.0 / frequency_hz:
        summary["reject_reason"] = "steady_state_shorter_than_two_cycles"
        return summary, None

    cmd_grid = resample_to_grid(cmd_t, cmd_y, grid)
    out_grid = resample_to_grid(odom_t, output_y, grid)
    if cmd_grid is None or out_grid is None:
        summary["reject_reason"] = "resampling_failed"
        return summary, None

    cmd_fit = fit_sine(grid, cmd_grid, omega)
    out_fit = fit_sine(grid, out_grid, omega)
    if cmd_fit is None or out_fit is None:
        summary["reject_reason"] = "sine_fit_failed"
        return summary, None
    if cmd_fit.amplitude <= 1e-6:
        summary["reject_reason"] = "input_amplitude_too_small"
        return summary, None

    output_amplitude = out_fit.amplitude
    if output_source == "odom_position_x":
        output_amplitude = omega * output_amplitude

    gain = output_amplitude / cmd_fit.amplitude
    gain_db = 20.0 * math.log10(gain) if gain > 0.0 else math.nan

    summary.update(
        {
            "A_in_fit": cmd_fit.amplitude,
            "A_out_fit": output_amplitude,
            "gain_linear": gain,
            "gain_db": gain_db,
            "phase_rad_optional": wrap_angle(out_fit.phase_rad - cmd_fit.phase_rad),
            "cmd_fit_rmse": cmd_fit.rmse,
            "cmd_fit_r2": cmd_fit.r2,
            "out_fit_rmse": out_fit.rmse,
            "out_fit_r2": out_fit.r2,
            "offset": out_fit.offset,
            "drift_rate": out_fit.drift_rate,
            "valid": bool(math.isfinite(gain) and gain >= 0.0),
            "reject_reason": "",
        }
    )

    if not summary["valid"]:
        summary["reject_reason"] = "nonfinite_gain"
        return summary, None

    return summary, RunAnalysisData(
        run_id=run_id,
        plot_id=plot_id,
        frequency_hz=frequency_hz,
        omega_rad_s=omega,
        t=grid,
        cmd=cmd_grid,
        output=out_grid,
        cmd_fit=cmd_fit.fitted,
        output_fit=out_fit.fitted,
        output_source=output_source,
    )


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_cell(row.get(field)) for field in SUMMARY_FIELDS})


def format_cell(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0.0 else "-inf"
        return f"{value:.9g}"
    return value


def import_pyplot() -> tuple[Optional[Any], dict[str, Any]]:
    plt, reason = try_import_pyplot()
    if plt is not None:
        return plt, {"mode": "default"}

    original_path = list(sys.path)
    removed_modules = remove_matplotlib_numpy_modules()
    remove_user_site_from_path()
    plt, fallback_reason = try_import_pyplot()
    if plt is not None:
        return plt, {
            "mode": "without_user_site",
            "initial_error": reason,
        }

    sys.path = original_path
    sys.modules.update(removed_modules)
    return None, {
        "mode": "failed",
        "initial_error": reason,
        "fallback_error": fallback_reason,
    }


def try_import_pyplot() -> tuple[Optional[Any], str]:
    stderr_capture = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr_capture):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        return plt, ""
    except Exception as exc:
        stderr_text = stderr_capture.getvalue().strip()
        reason = str(exc) or stderr_text or exc.__class__.__name__
        return None, reason


def remove_user_site_from_path() -> None:
    user_sites: set[str] = set()
    for value in (site.USER_SITE, site.getusersitepackages()):
        if not value:
            continue
        if isinstance(value, str):
            user_sites.add(str(Path(value).resolve()))
        else:
            user_sites.update(str(Path(path).resolve()) for path in value)

    sys.path = [
        path
        for path in sys.path
        if not path or str(Path(path).resolve()) not in user_sites
    ]


def remove_matplotlib_numpy_modules() -> dict[str, Any]:
    removed = {}
    for name in list(sys.modules):
        if name == "matplotlib" or name.startswith("matplotlib."):
            removed[name] = sys.modules.pop(name)
        elif name == "numpy" or name.startswith("numpy."):
            removed[name] = sys.modules.pop(name)
    return removed


def make_plots(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    analysis_data: list[RunAnalysisData],
) -> dict[str, Any]:
    plt, import_status = import_pyplot()
    if plt is None:
        return {
            "generated": False,
            "reason": "matplotlib import failed",
            "matplotlib_import": import_status,
        }

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    for data in analysis_data:
        time_path = plots_dir / f"time_response_{data.plot_id}.png"
        fig, ax = plt.subplots()
        ax.plot(data.t, data.cmd, label="command vx")
        ax.plot(data.t, data.output, label=data.output_source)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("velocity [m/s]" if data.output_source == "odom_twist_vx" else "position [m]")
        ax.set_title(f"time response {data.frequency_hz:.3f} Hz")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(time_path, dpi=150)
        plt.close(fig)
        files.append(str(time_path.relative_to(output_dir)))

        fit_path = plots_dir / f"sine_fit_{data.plot_id}.png"
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0].plot(data.t, data.cmd, label="cmd")
        axes[0].plot(data.t, data.cmd_fit, label="cmd fit")
        axes[0].set_ylabel("cmd vx [m/s]")
        axes[0].grid(True)
        axes[0].legend()
        axes[1].plot(data.t, data.output, label=data.output_source)
        axes[1].plot(data.t, data.output_fit, label="output fit")
        axes[1].set_xlabel("time [s]")
        axes[1].set_ylabel("output")
        axes[1].grid(True)
        axes[1].legend()
        fig.suptitle(f"sine fit {data.frequency_hz:.3f} Hz")
        fig.tight_layout()
        fig.savefig(fit_path, dpi=150)
        plt.close(fig)
        files.append(str(fit_path.relative_to(output_dir)))

    valid = [
        row
        for row in summaries
        if bool_from_metadata(row.get("valid"), default=False)
        and math.isfinite(float_or_nan(row.get("gain_linear")))
    ]
    if valid:
        valid.sort(key=lambda row: float_or_nan(row.get("f_hz")))
        frequencies = np.array([float_or_nan(row.get("f_hz")) for row in valid])
        gains = np.array([float_or_nan(row.get("gain_linear")) for row in valid])
        gains_db = np.array([float_or_nan(row.get("gain_db")) for row in valid])
        cmd_r2 = np.array([float_or_nan(row.get("cmd_fit_r2")) for row in valid])
        out_r2 = np.array([float_or_nan(row.get("out_fit_r2")) for row in valid])

        linear_path = plots_dir / "amplitude_ratio_linear.png"
        fig, ax = plt.subplots()
        ax.plot(frequencies, gains, marker="o")
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("gain [-]")
        ax.set_title("amplitude ratio")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(linear_path, dpi=150)
        plt.close(fig)
        files.append(str(linear_path.relative_to(output_dir)))

        db_path = plots_dir / "amplitude_ratio_db.png"
        fig, ax = plt.subplots()
        ax.plot(frequencies, gains_db, marker="o")
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("gain [dB]")
        ax.set_title("amplitude ratio dB")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(db_path, dpi=150)
        plt.close(fig)
        files.append(str(db_path.relative_to(output_dir)))

        quality_path = plots_dir / "fit_quality.png"
        fig, ax = plt.subplots()
        ax.plot(frequencies, cmd_r2, marker="o", label="cmd R2")
        ax.plot(frequencies, out_r2, marker="o", label="output R2")
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("R2 [-]")
        ax.set_title("fit quality")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(quality_path, dpi=150)
        plt.close(fig)
        files.append(str(quality_path.relative_to(output_dir)))

    return {
        "generated": True,
        "matplotlib_import": import_status,
        "files": files,
    }


def write_analysis_metadata(
    output_dir: Path,
    input_dirs: list[Path],
    source_experiment_dirs: list[Path],
    run_dirs: list[Path],
    summaries: list[dict[str, Any]],
    plot_status: dict[str, Any],
) -> None:
    valid_count = sum(1 for row in summaries if bool_from_metadata(row.get("valid"), default=False))
    metadata = {
        "summary_csv": str(output_dir / "frequency_summary.csv"),
        "output_dir": str(output_dir),
        "input_dirs": [str(path) for path in input_dirs],
        "source_experiment_dirs": [str(path) for path in source_experiment_dirs],
        "source_run_dirs": [str(path) for path in run_dirs],
        "run_count": len(summaries),
        "valid_run_count": valid_count,
        "plot_status": plot_status,
    }
    path = output_dir / "analysis_metadata.json"
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def collect_experiment_run_sources(experiment_dir: Path) -> list[tuple[Path, Path]]:
    runs_dir = experiment_dir / "runs"
    run_dirs = sorted(
        [path for path in runs_dir.iterdir() if path.is_dir()],
        key=run_sort_key,
    )
    if not run_dirs:
        raise SystemExit(f"no run directories found under: {runs_dir}")
    return [(experiment_dir, run_dir) for run_dir in run_dirs]


def discover_child_experiment_dirs(group_dir: Path) -> list[Path]:
    if not group_dir.is_dir():
        return []
    return sorted(
        [
            path
            for path in group_dir.iterdir()
            if path.is_dir() and (path / "runs").is_dir()
        ],
        key=lambda path: path.name,
    )


def collect_run_sources(experiment_dirs: list[Path]) -> list[tuple[Path, Path]]:
    run_sources: list[tuple[Path, Path]] = []
    for experiment_dir in experiment_dirs:
        runs_dir = experiment_dir / "runs"
        if runs_dir.is_dir():
            run_sources.extend(collect_experiment_run_sources(experiment_dir))
            continue

        if (experiment_dir / "run_metadata.json").is_file():
            source_experiment_dir = experiment_dir.parent.parent
            run_sources.append((source_experiment_dir, experiment_dir))
            continue

        child_experiment_dirs = discover_child_experiment_dirs(experiment_dir)
        if child_experiment_dirs:
            for child_experiment_dir in child_experiment_dirs:
                run_sources.extend(collect_experiment_run_sources(child_experiment_dir))
            continue

        raise SystemExit(
            f"no runs directory or child experiment directories found under: {experiment_dir}"
        )

    return sorted(run_sources, key=lambda item: (*run_sort_key(item[1]), str(item[0])))


def source_experiment_dirs(run_sources: list[tuple[Path, Path]]) -> list[Path]:
    return list(dict.fromkeys(source_experiment_dir for source_experiment_dir, _ in run_sources))


def default_output_dir(input_dirs: list[Path], discovered_source_experiment_dirs: list[Path]) -> Path:
    if len(input_dirs) == 1:
        input_dir = input_dirs[0]
        if (
            not (input_dir / "runs").is_dir()
            and not (input_dir / "run_metadata.json").is_file()
            and discovered_source_experiment_dirs
        ):
            return input_dir / "analysis"
        return input_dir

    parent = input_dirs[0].parent
    if any(path.parent != parent for path in input_dirs):
        parent = Path.cwd()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return parent / f"{timestamp}_merged_frequency_response"


def make_plot_ids(run_sources: list[tuple[Path, Path]]) -> dict[Path, str]:
    name_counts: dict[str, int] = {}
    for _, run_dir in run_sources:
        name_counts[run_dir.name] = name_counts.get(run_dir.name, 0) + 1

    plot_ids: dict[Path, str] = {}
    for source_experiment_dir, run_dir in run_sources:
        if name_counts[run_dir.name] == 1:
            plot_ids[run_dir] = sanitize_token(run_dir.name)
        else:
            plot_ids[run_dir] = sanitize_token(f"{source_experiment_dir.name}_{run_dir.name}")
    return plot_ids


def analyze_frequency_response(
    experiment_dirs: list[Path],
    *,
    output_dir: Optional[Path] = None,
    resample_rate: float = 100.0,
    default_discard_cycles: float = 2.0,
    plots: bool = True,
) -> dict[str, Any]:
    """Analyze one or more frequency-response experiment directories."""
    input_dirs = [Path(path) for path in experiment_dirs]
    run_sources = collect_run_sources(input_dirs)
    discovered_source_experiment_dirs = source_experiment_dirs(run_sources)
    run_dirs = [run_dir for _, run_dir in run_sources]
    plot_ids = make_plot_ids(run_sources)
    resolved_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else default_output_dir(input_dirs, discovered_source_experiment_dirs)
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    options = FrequencyAnalysisOptions(
        resample_rate=float(resample_rate),
        default_discard_cycles=float(default_discard_cycles),
    )
    summaries: list[dict[str, Any]] = []
    analysis_data: list[RunAnalysisData] = []
    for source_experiment_dir, run_dir in run_sources:
        summary, data = analyze_run(run_dir, source_experiment_dir, plot_ids[run_dir], options)
        summaries.append(summary)
        if data is not None:
            analysis_data.append(data)

    summary_csv = resolved_output_dir / "frequency_summary.csv"
    write_summary(summary_csv, summaries)
    plot_status = (
        make_plots(resolved_output_dir, summaries, analysis_data)
        if plots
        else {"generated": False, "reason": "disabled"}
    )
    write_analysis_metadata(
        resolved_output_dir,
        input_dirs,
        discovered_source_experiment_dirs,
        run_dirs,
        summaries,
        plot_status,
    )

    valid_count = sum(1 for row in summaries if bool_from_metadata(row.get("valid"), default=False))
    return {
        "summary_csv": str(summary_csv),
        "output_dir": str(resolved_output_dir),
        "input_dirs": [str(path) for path in input_dirs],
        "source_experiment_dirs": [str(path) for path in discovered_source_experiment_dirs],
        "source_run_dirs": [str(path) for path in run_dirs],
        "run_count": len(summaries),
        "valid_run_count": valid_count,
        "plot_status": plot_status,
        "summaries": summaries,
    }


def parse_args(argv: Optional[list[str]] = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Analyze RB-KAIROS frequency-response recordings.")
    parser.add_argument(
        "--experiment-dir",
        action="append",
        nargs="+",
        required=True,
        help="one or more frequency_response experiment directories; can be repeated",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "analysis output directory; defaults to the input directory for one recorder "
            "experiment, <group>/analysis for one grouped directory, or a timestamped "
            "merged_frequency_response directory for multiple inputs"
        ),
    )
    parser.add_argument(
        "--resample-rate",
        type=float,
        default=100.0,
        help="common resampling rate [Hz]",
    )
    parser.add_argument(
        "--default-discard-cycles",
        type=float,
        default=2.0,
        help="fallback discard cycles when run metadata lacks the value",
    )
    args, ros_args = parser.parse_known_args(argv)
    args.experiment_dir = [
        path
        for experiment_dir_group in args.experiment_dir
        for path in experiment_dir_group
    ]
    if ros_args:
        if "--ros-args" not in ros_args:
            parser.error("unknown arguments: " + " ".join(ros_args))
        ros_arg_index = ros_args.index("--ros-args")
        if ros_arg_index > 0:
            parser.error("unknown arguments: " + " ".join(ros_args[:ros_arg_index]))
        ros_args = ros_args[ros_arg_index:]
    if args.resample_rate <= 0.0:
        parser.error("--resample-rate must be > 0")
    if args.default_discard_cycles < 0.0:
        parser.error("--default-discard-cycles must be >= 0")
    return args, ros_args


def main(argv: Optional[list[str]] = None) -> None:
    args, _ = parse_args(argv)
    result = analyze_frequency_response(
        [Path(path) for path in args.experiment_dir],
        output_dir=Path(args.output_dir) if args.output_dir is not None else None,
        resample_rate=float(args.resample_rate),
        default_discard_cycles=float(args.default_discard_cycles),
        plots=True,
    )
    print(f"Analyzed {result['run_count']} runs, valid={result['valid_run_count']}")
    print(f"Summary: {result['summary_csv']}")
    print(f"Plots: {Path(str(result['output_dir'])) / 'plots'}")


if __name__ == "__main__":
    main()
