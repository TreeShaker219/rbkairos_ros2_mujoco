#!/usr/bin/env python3

import argparse
import csv
import json
import math
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

np = None
plt = None


SUMMARY_FIELDS = [
    "run_id",
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
    "phase_rad",
    "phase_deg",
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
    "output_source",
    "valid",
    "reject_reason",
]


def read_csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def load_numpy():
    global np
    if np is None:
        import numpy as numpy_module
        np = numpy_module
    return np


def load_matplotlib():
    global plt
    if plt is not None:
        return True
    if matplotlib_numpy_abi_risk():
        print(
            "warning: matplotlib appears incompatible with the installed NumPy, skipping plots",
            file=sys.stderr,
        )
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot_module
    except Exception as exc:
        print(f"warning: matplotlib unavailable, skipping plots: {exc}", file=sys.stderr)
        return False
    plt = pyplot_module
    return True


def parse_version_prefix(version_text):
    parts = []
    for token in version_text.split("."):
        digits = ""
        for char in token:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def matplotlib_numpy_abi_risk():
    if np is None:
        return False
    numpy_version = parse_version_prefix(getattr(np, "__version__", "0"))
    if not numpy_version or numpy_version[0] < 2:
        return False
    try:
        matplotlib_version = parse_version_prefix(importlib_metadata.version("matplotlib"))
    except importlib_metadata.PackageNotFoundError:
        return False
    return matplotlib_version < (3, 8)


def to_float_array(rows, key):
    values = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            values.append(math.nan)
    return np.asarray(values, dtype=float)


def wrap_phase(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def fit_sine(t, y, omega):
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if t.size < 4:
        return None

    design = np.column_stack([
        np.ones_like(t),
        t,
        np.sin(omega * t),
        np.cos(omega * t),
    ])
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coeffs
    residual = y - fitted
    rmse = float(np.sqrt(np.mean(residual * residual)))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum(residual * residual) / denom) if denom > 0.0 else math.nan
    a = float(coeffs[2])
    b = float(coeffs[3])
    amplitude = float(math.hypot(a, b))
    phase = float(math.atan2(b, a))
    return {
        "offset": float(coeffs[0]),
        "drift_rate": float(coeffs[1]),
        "a": a,
        "b": b,
        "amplitude": amplitude,
        "phase": phase,
        "rmse": rmse,
        "r2": r2,
        "sample_count": int(t.size),
    }


def collect_run_dirs(experiment_dir):
    direct_runs = sorted((experiment_dir / "runs").glob("f_*Hz"))
    if direct_runs:
        return [path for path in direct_runs if (path / "cmd_vel.csv").exists()]

    nested = []
    for runs_dir in sorted(experiment_dir.glob("*/runs")):
        nested.extend(sorted(path for path in runs_dir.glob("f_*Hz") if (path / "cmd_vel.csv").exists()))
    return nested


def select_output_source(args, odom_rows):
    vx = to_float_array(odom_rows, "vx")
    x = to_float_array(odom_rows, "x")
    if args.output_source == "odom_twist_vx":
        return "odom_twist_vx", vx
    if args.output_source == "odom_position_x":
        return "odom_position_x", x
    finite_vx = vx[np.isfinite(vx)]
    if finite_vx.size > 0 and float(np.nanstd(finite_vx)) > 1e-8:
        return "odom_twist_vx", vx
    return "odom_position_x", x


def analyze_run(args, experiment_dir, run_dir):
    cmd_rows = read_csv_rows(run_dir / "cmd_vel.csv")
    odom_rows = read_csv_rows(run_dir / "odom.csv")
    metadata_path = run_dir / "run_metadata.json"
    run_metadata = read_json(metadata_path) if metadata_path.exists() else {}

    if not cmd_rows or not odom_rows:
        return invalid_summary(experiment_dir, run_dir, run_metadata, "missing cmd or odom samples")

    frequency_hz = float(run_metadata.get("frequency_hz", cmd_rows[0]["frequency_hz"]))
    omega = 2.0 * math.pi * frequency_hz
    cycles = float(run_metadata.get("cycles", args.cycles))
    discard_cycles = float(run_metadata.get("discard_cycles", args.discard_cycles))
    ramp_cycles = float(run_metadata.get("ramp_cycles", args.ramp_cycles))
    analysis_start_s = max(discard_cycles / frequency_hz, ramp_cycles / frequency_hz)
    min_duration_s = args.min_analysis_cycles / frequency_hz

    t_cmd = to_float_array(cmd_rows, "t_experiment")
    u = to_float_array(cmd_rows, "cmd_vx")
    cmd_mask = np.isfinite(t_cmd) & (t_cmd >= analysis_start_s)
    t_cmd_fit = t_cmd[cmd_mask]
    u_fit = u[cmd_mask]

    output_source, output_values = select_output_source(args, odom_rows)
    t_out = to_float_array(odom_rows, "t_experiment")
    out_mask = np.isfinite(t_out) & (t_out >= analysis_start_s)
    t_out_fit = t_out[out_mask]
    y_fit_values = output_values[out_mask]

    cmd_fit = fit_sine(t_cmd_fit, u_fit, omega)
    out_fit = fit_sine(t_out_fit, y_fit_values, omega)
    if cmd_fit is None or out_fit is None:
        return invalid_summary(experiment_dir, run_dir, run_metadata, "not enough finite samples")

    output_amplitude = out_fit["amplitude"]
    output_phase = out_fit["phase"]
    if output_source == "odom_position_x":
        output_amplitude = omega * output_amplitude
        output_phase = wrap_phase(output_phase + 0.5 * math.pi)

    gain = output_amplitude / cmd_fit["amplitude"] if cmd_fit["amplitude"] > 0.0 else math.nan
    gain_db = 20.0 * math.log10(gain) if gain > 0.0 and math.isfinite(gain) else math.nan
    phase = wrap_phase(output_phase - cmd_fit["phase"])

    x = to_float_array(odom_rows, "x")
    y = to_float_array(odom_rows, "y")
    yaw = to_float_array(odom_rows, "yaw")
    final_dx = finite_delta(x)
    final_dy = finite_delta(y)
    final_dyaw = wrap_phase(finite_delta(yaw))
    max_abs_x = finite_max_abs(x - first_finite(x))
    max_abs_y = finite_max_abs(y - first_finite(y))
    max_abs_yaw = finite_max_abs(np.asarray([wrap_phase(value - first_finite(yaw)) for value in yaw]))

    reject_reasons = []
    if cmd_fit["r2"] < args.min_cmd_r2:
        reject_reasons.append(f"cmd_fit_r2<{args.min_cmd_r2}")
    if not math.isfinite(out_fit["r2"]):
        reject_reasons.append("out_fit_r2_not_finite")
    elif out_fit["r2"] < args.min_out_r2:
        reject_reasons.append(f"out_fit_r2<{args.min_out_r2}")
    if not math.isfinite(gain) or not math.isfinite(gain_db) or not math.isfinite(phase):
        reject_reasons.append("nonfinite_gain_or_phase")
    if cmd_fit["amplitude"] <= 0.0:
        reject_reasons.append("zero_input_amplitude")
    if t_cmd_fit.size == 0 or (np.nanmax(t_cmd_fit) - np.nanmin(t_cmd_fit)) < min_duration_s:
        reject_reasons.append("analysis_window_too_short")
    if run_metadata.get("stop_reason", "completed") != "completed":
        reject_reasons.append(f"stop_reason={run_metadata.get('stop_reason')}")

    return {
        "run_id": run_dir.name,
        "source_experiment_dir": str(experiment_dir),
        "source_run_dir": str(run_dir),
        "environment": args.environment,
        "axis": run_metadata.get("axis", "x"),
        "f_hz": frequency_hz,
        "omega_rad_s": omega,
        "A_cmd_nominal": run_metadata.get("amplitude", math.nan),
        "A_in_fit": cmd_fit["amplitude"],
        "A_out_fit": output_amplitude,
        "gain_linear": gain,
        "gain_db": gain_db,
        "phase_rad": phase,
        "phase_deg": math.degrees(phase) if math.isfinite(phase) else math.nan,
        "cmd_fit_rmse": cmd_fit["rmse"],
        "cmd_fit_r2": cmd_fit["r2"],
        "out_fit_rmse": out_fit["rmse"],
        "out_fit_r2": out_fit["r2"],
        "offset": out_fit["offset"],
        "drift_rate": out_fit["drift_rate"],
        "final_dx": final_dx,
        "final_dy": final_dy,
        "final_dyaw": final_dyaw,
        "max_abs_x": max_abs_x,
        "max_abs_y": max_abs_y,
        "max_abs_yaw": max_abs_yaw,
        "output_source": output_source,
        "valid": len(reject_reasons) == 0,
        "reject_reason": ";".join(reject_reasons),
        "_plot": {
            "t_cmd": t_cmd,
            "u": u,
            "t_out": t_out,
            "output": output_values,
            "output_source": output_source,
        },
    }


def invalid_summary(experiment_dir, run_dir, run_metadata, reason):
    frequency_hz = float(run_metadata.get("frequency_hz", math.nan))
    omega = 2.0 * math.pi * frequency_hz if math.isfinite(frequency_hz) else math.nan
    row = {field: math.nan for field in SUMMARY_FIELDS}
    row.update({
        "run_id": run_dir.name,
        "source_experiment_dir": str(experiment_dir),
        "source_run_dir": str(run_dir),
        "axis": run_metadata.get("axis", "x"),
        "f_hz": frequency_hz,
        "omega_rad_s": omega,
        "valid": False,
        "reject_reason": reason,
    })
    return row


def first_finite(values):
    finite = values[np.isfinite(values)]
    return float(finite[0]) if finite.size else math.nan


def finite_delta(values):
    finite = values[np.isfinite(values)]
    return float(finite[-1] - finite[0]) if finite.size >= 2 else math.nan


def finite_max_abs(values):
    finite = values[np.isfinite(values)]
    return float(np.max(np.abs(finite))) if finite.size else math.nan


def write_summary(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def write_plots(analysis_dir, rows):
    valid_rows = [row for row in rows if row.get("valid") is True]
    if valid_rows:
        freqs = np.asarray([row["f_hz"] for row in valid_rows], dtype=float)
        gains = np.asarray([row["gain_db"] for row in valid_rows], dtype=float)
        phases = np.asarray([row["phase_deg"] for row in valid_rows], dtype=float)
        order = np.argsort(freqs)

        fig, (ax_gain, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
        ax_gain.plot(freqs[order], gains[order], marker="o")
        ax_gain.set_ylabel("Gain (dB)")
        ax_gain.grid(True)
        ax_phase.plot(freqs[order], phases[order], marker="o")
        ax_phase.set_xlabel("Frequency (Hz)")
        ax_phase.set_ylabel("Phase (deg)")
        ax_phase.grid(True)
        fig.tight_layout()
        fig.savefig(analysis_dir / "bode_gain_phase.png", dpi=160)
        plt.close(fig)

    for row in rows:
        plot_data = row.get("_plot")
        if not plot_data:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(plot_data["t_cmd"], plot_data["u"], label="cmd_vx")
        ax.plot(plot_data["t_out"], plot_data["output"], label=plot_data["output_source"])
        ax.set_xlabel("t_experiment (s)")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(analysis_dir / f"time_series_{row['run_id']}.png", dpi=160)
        plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze RBKairos real frequency response CSV output.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--environment", default="real_lab")
    parser.add_argument("--output-source", choices=["auto", "odom_twist_vx", "odom_position_x"], default="auto")
    parser.add_argument("--cycles", type=float, default=6.0)
    parser.add_argument("--discard-cycles", type=float, default=2.0)
    parser.add_argument("--ramp-cycles", type=float, default=1.0)
    parser.add_argument("--min-analysis-cycles", type=float, default=2.0)
    parser.add_argument("--min-cmd-r2", type=float, default=0.95)
    parser.add_argument("--min-out-r2", type=float, default=0.70)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    experiment_dir = Path(args.experiment_dir)
    run_dirs = collect_run_dirs(experiment_dir)
    if not run_dirs:
        print(f"No run directories found in {experiment_dir}", file=sys.stderr)
        return 1

    try:
        load_numpy()
    except Exception as exc:
        print(f"numpy is required for analysis: {exc}", file=sys.stderr)
        return 1

    analysis_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows = [analyze_run(args, experiment_dir, run_dir) for run_dir in run_dirs]
    write_summary(analysis_dir / "frequency_summary.csv", rows)
    if not args.skip_plots and load_matplotlib():
        write_plots(analysis_dir, rows)

    for row in rows:
        row.pop("_plot", None)
    with (analysis_dir / "analysis_metadata.json").open("w") as handle:
        json.dump({"experiment_dir": str(experiment_dir), "run_count": len(rows)}, handle, indent=2)
        handle.write("\n")

    valid_count = sum(1 for row in rows if row.get("valid") is True)
    print(f"Analyzed {len(rows)} runs, valid={valid_count}")
    print(f"Summary: {analysis_dir / 'frequency_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
