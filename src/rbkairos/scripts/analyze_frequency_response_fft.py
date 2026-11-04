#!/usr/bin/env python3

"""FFT-based analysis for RB-KAIROS frequency-response experiments.

This script reuses the same recorded CSV files as ``analyze_frequency_response.py``,
writes to a separate ``fft_analysis`` directory by default, and reports only
FFT-derived gain, phase, spectrum, sampling, and harmonic diagnostics.

Assumptions documented in the output metadata:

* input is velocity command ``cmd_vel.csv:cmd_vx``;
* default output is ``odom.csv:vx`` selected by the same auto rule as the
  non-FFT real-robot analyzer;
* FFT is only applied after interpolation to a uniform, integer-cycle window;
* no detrending is applied before FFT beyond mean removal;
* harmonic validity is limited by the raw command/output sampling rates and
  the chosen uniform-grid Nyquist frequency.
"""

import argparse
import csv
import html
import json
import math
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

np = None
plt = None


FFT_SUMMARY_FIELDS = [
    "run_id",
    "source_experiment_dir",
    "source_run_dir",
    "environment",
    "axis",
    "output_source",
    "f_hz",
    "omega_rad_s",
    "A_cmd_nominal",
    "analysis_start_s",
    "analysis_end_s",
    "analysis_duration_s",
    "analysis_cycles",
    "grid_sample_count",
    "grid_rate_hz",
    "fundamental_bin",
    "fft_gain_linear",
    "fft_gain_db",
    "fft_phase_deg",
    "fft_input_amp",
    "fft_output_amp",
    "fft_input_phase_deg",
    "fft_output_phase_deg",
    "fft_output_fundamental_energy_fraction",
    "fft_output_noise_floor_amp",
    "thd_2_to_max",
    "odd_thd_3_5_7",
    "valid",
    "reject_reason",
]


HARMONIC_FIELDS = [
    "run_id",
    "source_run_dir",
    "f_hz",
    "harmonic",
    "harmonic_frequency_hz",
    "bin_index",
    "nyquist_valid",
    "input_amp",
    "output_amp",
    "input_amp_relative_to_fundamental",
    "output_amp_relative_to_fundamental",
    "output_energy_fraction",
    "output_noise_floor_amp",
    "output_snr_db",
    "thd_2_to_max",
    "odd_thd_3_5_7",
    "valid",
    "reject_reason",
]


SAMPLING_FIELDS = [
    "run_id",
    "source_run_dir",
    "f_hz",
    "cmd_raw_sample_count",
    "cmd_finite_pair_count",
    "cmd_nonmonotonic_count",
    "cmd_duplicate_time_count",
    "cmd_t_start_s",
    "cmd_t_end_s",
    "cmd_duration_s",
    "cmd_median_dt_s",
    "cmd_min_dt_s",
    "cmd_max_dt_s",
    "cmd_dt_cv",
    "cmd_median_rate_hz",
    "odom_raw_sample_count",
    "odom_finite_pair_count",
    "odom_nonmonotonic_count",
    "odom_duplicate_time_count",
    "odom_t_start_s",
    "odom_t_end_s",
    "odom_duration_s",
    "odom_median_dt_s",
    "odom_min_dt_s",
    "odom_max_dt_s",
    "odom_dt_cv",
    "odom_median_rate_hz",
    "raw_min_nyquist_hz",
    "grid_rate_hz",
    "grid_nyquist_hz",
    "nyquist_safety_factor",
    "max_valid_harmonic_by_nyquist",
    "analysis_start_s",
    "analysis_end_s",
    "analysis_duration_s",
    "analysis_cycles",
    "grid_sample_count",
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
            "warning: matplotlib appears incompatible with the installed NumPy, using SVG plot fallback",
            file=sys.stderr,
        )
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot_module
    except Exception as exc:
        print(f"warning: matplotlib unavailable, using SVG plot fallback: {exc}", file=sys.stderr)
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


def safe_float(value, default=math.nan):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_float_array(rows, key):
    values = []
    for row in rows:
        values.append(safe_float(row.get(key)))
    return np.asarray(values, dtype=float)


def wrap_phase_rad(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def gain_db(gain):
    return 20.0 * math.log10(gain) if gain > 0.0 and math.isfinite(gain) else math.nan


def csv_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fields})


def svg_text(value):
    return html.escape(str(value), quote=True)


def chart_extent(values, pad_fraction=0.08):
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None
    lower = min(clean)
    upper = max(clean)
    if lower == upper:
        padding = abs(lower if lower else 1.0) * 0.1
        lower -= padding
        upper += padding
    padding = (upper - lower) * pad_fraction
    return lower - padding, upper + padding


def format_tick(value):
    if not math.isfinite(value):
        return ""
    return f"{value:.4g}"


def write_svg_line_chart(
    path,
    title,
    x_label,
    y_label,
    series,
    *,
    log_x=True,
    log_y=False,
    width=980,
    height=330,
):
    margin = {"left": 72.0, "right": 24.0, "top": 36.0, "bottom": 56.0}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    prepared = []
    transformed_x = []
    transformed_y = []
    raw_x_values = []
    for label, color, x_values, y_values in series:
        points = []
        for x_raw, y_raw in zip(x_values, y_values):
            x = safe_float(x_raw)
            y = safe_float(y_raw)
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            if log_x and x <= 0.0:
                continue
            if log_y and y <= 0.0:
                continue
            x_plot = math.log10(x) if log_x else x
            y_plot = math.log10(y) if log_y else y
            points.append((x, y, x_plot, y_plot))
            transformed_x.append(x_plot)
            transformed_y.append(y_plot)
            raw_x_values.append(x)
        if points:
            prepared.append((label, color, points))

    x_extent = chart_extent(transformed_x, 0.04)
    y_extent = chart_extent(transformed_y, 0.10)
    if not prepared or x_extent is None or y_extent is None:
        return None

    x_min, x_max = x_extent
    y_min, y_max = y_extent

    def sx(x_plot):
        return margin["left"] + (x_plot - x_min) / (x_max - x_min) * plot_width

    def sy(y_plot):
        return margin["top"] + plot_height - (y_plot - y_min) / (y_max - y_min) * plot_height

    def line(x1, y1, x2, y2, stroke="#e5e7eb", stroke_width=1.0):
        return (
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.2f}"/>'
        )

    def text(x, y, value, anchor="middle", size=12, color="#6b7280", extra=""):
        return (
            f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
            f'font-family="sans-serif" font-size="{size}" fill="{color}"{extra}>'
            f'{svg_text(value)}</text>'
        )

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2.0, 22.0, title, size=15, color="#1f2937"),
        line(margin["left"], margin["top"], margin["left"], margin["top"] + plot_height, "#9ca3af", 1.2),
        line(margin["left"], margin["top"] + plot_height, margin["left"] + plot_width, margin["top"] + plot_height, "#9ca3af", 1.2),
    ]

    unique_x_ticks = sorted({float(f"{value:.6g}") for value in raw_x_values})
    if len(unique_x_ticks) <= 12:
        x_ticks = unique_x_ticks
    else:
        x_ticks = [
            10.0 ** (x_min + (x_max - x_min) * index / 6.0)
            if log_x
            else x_min + (x_max - x_min) * index / 6.0
            for index in range(7)
        ]
    for tick in x_ticks:
        x_plot = math.log10(tick) if log_x else tick
        if x_plot < x_min or x_plot > x_max:
            continue
        x_coord = sx(x_plot)
        elements.append(line(x_coord, margin["top"], x_coord, margin["top"] + plot_height, "#eef2f7"))
        elements.append(text(x_coord, margin["top"] + plot_height + 20.0, format_tick(tick), size=11))

    for index in range(6):
        y_plot = y_min + (y_max - y_min) * index / 5.0
        y_coord = sy(y_plot)
        label_value = 10.0 ** y_plot if log_y else y_plot
        elements.append(line(margin["left"], y_coord, margin["left"] + plot_width, y_coord, "#eef2f7"))
        elements.append(text(margin["left"] - 9.0, y_coord + 4.0, format_tick(label_value), anchor="end", size=11))

    elements.append(text(width / 2.0, height - 14.0, x_label, size=12))
    elements.append(
        text(
            18.0,
            height / 2.0,
            y_label,
            size=12,
            extra=f' transform="rotate(-90 18 {height / 2.0:.2f})"',
        )
    )

    legend_x = margin["left"] + 8.0
    legend_y = margin["top"] + 16.0
    for series_index, (label, color, points) in enumerate(prepared):
        path_data = " ".join(
            f"{'M' if point_index == 0 else 'L'} {sx(x_plot):.2f} {sy(y_plot):.2f}"
            for point_index, (_, _, x_plot, y_plot) in enumerate(points)
        )
        elements.append(
            f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2.2"/>'
        )
        for x_raw, y_raw, x_plot, y_plot in points:
            elements.append(
                f'<circle cx="{sx(x_plot):.2f}" cy="{sy(y_plot):.2f}" r="3.4" fill="{color}">'
                f'<title>{svg_text(label)}\nf={format_tick(x_raw)} Hz\ny={format_tick(y_raw)}</title>'
                "</circle>"
            )
        y_legend = legend_y + 18.0 * series_index
        elements.append(line(legend_x, y_legend - 4.0, legend_x + 18.0, y_legend - 4.0, color, 2.2))
        elements.append(text(legend_x + 24.0, y_legend, label, anchor="start", size=11, color="#374151"))

    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")
    return path


def write_svg_plots(analysis_dir, summary_rows, harmonic_rows, spectra, args):
    written = []
    valid_rows = [row for row in summary_rows if row.get("valid") is True]
    valid_rows.sort(key=lambda row: safe_float(row.get("f_hz")))

    if valid_rows:
        freqs = [safe_float(row.get("f_hz")) for row in valid_rows]
        gain_db_values = [safe_float(row.get("fft_gain_db")) for row in valid_rows]
        path = write_svg_line_chart(
            analysis_dir / "gain_fft.svg",
            "FFT fundamental gain",
            "Excitation frequency (Hz)",
            "Gain (dB)",
            [("FFT fundamental", "#dc2626", freqs, gain_db_values)],
        )
        if path is not None:
            written.append(str(path))

        phase_values = [safe_float(row.get("fft_phase_deg")) for row in valid_rows]
        path = write_svg_line_chart(
            analysis_dir / "phase_fft.svg",
            "FFT fundamental phase",
            "Excitation frequency (Hz)",
            "Phase (deg)",
            [("FFT fundamental", "#dc2626", freqs, phase_values)],
        )
        if path is not None:
            written.append(str(path))

    harmonic_series = []
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    for harmonic in range(2, args.max_harmonic + 1):
        rows = [
            row for row in harmonic_rows
            if int(row.get("harmonic", -1)) == harmonic and row.get("nyquist_valid") is True
        ]
        rows.sort(key=lambda row: safe_float(row.get("f_hz")))
        if not rows:
            continue
        x_values = [safe_float(row.get("f_hz")) for row in rows]
        y_values = [
            100.0 * safe_float(row.get("output_amp_relative_to_fundamental"))
            for row in rows
        ]
        harmonic_series.append(
            (f"H{harmonic}", colors[(harmonic - 2) % len(colors)], x_values, y_values)
        )
    if harmonic_series:
        path = write_svg_line_chart(
            analysis_dir / "harmonic_magnitudes.svg",
            "Output harmonic / fundamental",
            "Excitation frequency (Hz)",
            "Output harmonic / fundamental (%)",
            harmonic_series,
        )
        if path is not None:
            written.append(str(path))

    spectra_to_plot = [spectrum for spectrum in spectra if spectrum and spectrum.get("valid")]
    spectra_to_plot.sort(key=lambda item: safe_float(item.get("f_hz")))
    for spectrum in spectra_to_plot[: max(0, args.max_spectrum_plots)]:
        freqs = spectrum["freqs"]
        input_amplitudes = spectrum["input_amplitudes"]
        output_amplitudes = spectrum["output_amplitudes"]
        max_freq = spectrum.get("nyquist_limit_hz", math.nan)
        if not math.isfinite(max_freq) or max_freq <= 0.0:
            max_freq = float(freqs[-1])
        mask = (freqs > 0.0) & (freqs <= max_freq)
        if not np.any(mask):
            continue
        path = write_svg_line_chart(
            analysis_dir / f"spectra_{spectrum['run_id']}.svg",
            f"Spectrum {spectrum['run_id']}",
            "Frequency (Hz)",
            "Single-sided amplitude",
            [
                ("cmd_vx", "#2563eb", freqs[mask], input_amplitudes[mask]),
                ("output velocity", "#dc2626", freqs[mask], output_amplitudes[mask]),
            ],
            log_x=False,
            log_y=True,
            height=300,
        )
        if path is not None:
            written.append(str(path))

    return written


def collect_run_dirs(experiment_dir):
    if (experiment_dir / "cmd_vel.csv").exists() and (experiment_dir / "odom.csv").exists():
        return [experiment_dir]

    direct_runs = sorted((experiment_dir / "runs").glob("f_*Hz"))
    if direct_runs:
        return [path for path in direct_runs if (path / "cmd_vel.csv").exists()]

    nested = []
    for runs_dir in sorted(experiment_dir.rglob("runs")):
        nested.extend(sorted(path for path in runs_dir.glob("f_*Hz") if (path / "cmd_vel.csv").exists()))
    return nested


def source_experiment_dir_for_run(run_dir):
    if run_dir.parent.name == "runs":
        return run_dir.parent.parent
    return run_dir.parent


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


def sampling_stats(t, values):
    raw_count = int(t.size)
    finite_mask = np.isfinite(t) & np.isfinite(values)
    finite_t = t[finite_mask]
    finite_values = values[finite_mask]
    finite_pair_count = int(finite_t.size)

    row = {
        "raw_sample_count": raw_count,
        "finite_pair_count": finite_pair_count,
        "nonmonotonic_count": 0,
        "duplicate_time_count": 0,
        "t_start_s": math.nan,
        "t_end_s": math.nan,
        "duration_s": math.nan,
        "median_dt_s": math.nan,
        "min_dt_s": math.nan,
        "max_dt_s": math.nan,
        "dt_cv": math.nan,
        "median_rate_hz": math.nan,
    }
    if finite_pair_count == 0:
        return row

    diff_original = np.diff(finite_t)
    row["nonmonotonic_count"] = int(np.sum(diff_original <= 0.0)) if diff_original.size else 0

    order = np.argsort(finite_t, kind="mergesort")
    sorted_t = finite_t[order]
    sorted_values = finite_values[order]
    unique_t, inverse, counts = np.unique(sorted_t, return_inverse=True, return_counts=True)
    row["duplicate_time_count"] = int(finite_pair_count - unique_t.size)

    if unique_t.size == 0:
        return row

    row["t_start_s"] = float(unique_t[0])
    row["t_end_s"] = float(unique_t[-1])
    row["duration_s"] = float(unique_t[-1] - unique_t[0]) if unique_t.size >= 2 else 0.0

    if unique_t.size >= 2:
        dt = np.diff(unique_t)
        positive_dt = dt[dt > 0.0]
        if positive_dt.size:
            mean_dt = float(np.mean(positive_dt))
            row["median_dt_s"] = float(np.median(positive_dt))
            row["min_dt_s"] = float(np.min(positive_dt))
            row["max_dt_s"] = float(np.max(positive_dt))
            row["dt_cv"] = float(np.std(positive_dt) / mean_dt) if mean_dt > 0.0 else math.nan
            row["median_rate_hz"] = 1.0 / row["median_dt_s"] if row["median_dt_s"] > 0.0 else math.nan

    # Keep this function responsible only for diagnostics; values are averaged
    # in clean_timeseries() when interpolation is required.
    _ = sorted_values
    return row


def clean_timeseries(t, values):
    mask = np.isfinite(t) & np.isfinite(values)
    t = t[mask]
    values = values[mask]
    if t.size == 0:
        return t, values

    order = np.argsort(t, kind="mergesort")
    t = t[order]
    values = values[order]
    unique_t, inverse, counts = np.unique(t, return_inverse=True, return_counts=True)
    if unique_t.size == t.size:
        return t, values

    sums = np.bincount(inverse, weights=values)
    averaged_values = sums / counts
    return unique_t, averaged_values


def choose_integer_cycle_window(args, frequency_hz, t_cmd, t_out, discard_cycles, ramp_cycles):
    if frequency_hz <= 0.0 or not math.isfinite(frequency_hz):
        raise ValueError("invalid frequency")
    if t_cmd.size < 2 or t_out.size < 2:
        raise ValueError("not enough samples after finite filtering")

    requested_start = max(discard_cycles / frequency_hz, ramp_cycles / frequency_hz)
    common_start = max(requested_start, float(t_cmd[0]), float(t_out[0]))
    common_end = min(float(t_cmd[-1]), float(t_out[-1]))
    if common_end <= common_start:
        raise ValueError("no overlapping post-discard time window")

    available_cycles_float = (common_end - common_start) * frequency_hz
    available_cycles = int(math.floor(available_cycles_float + 1e-9))
    requested_cycles = int(args.analysis_cycles) if args.analysis_cycles else available_cycles
    if requested_cycles <= available_cycles:
        selected_cycles = requested_cycles
    elif available_cycles_float >= requested_cycles - args.cycle_shortfall_tolerance:
        shifted_start = common_end - requested_cycles / frequency_hz
        if shifted_start >= requested_start - args.cycle_shortfall_tolerance / frequency_hz:
            common_start = shifted_start
            selected_cycles = requested_cycles
        else:
            selected_cycles = available_cycles
    else:
        selected_cycles = available_cycles
    if selected_cycles < args.min_analysis_cycles:
        raise ValueError(
            f"analysis_window_too_short: available_cycles={available_cycles}, "
            f"min={args.min_analysis_cycles}"
        )

    duration = selected_cycles / frequency_hz
    end = common_start + duration
    if end > common_end + 1e-9:
        raise ValueError("integer-cycle window exceeds common data range")

    sample_count = int(round(args.grid_rate_hz * duration))
    sample_count = max(sample_count, 2 * selected_cycles + 1)
    if sample_count < 4:
        raise ValueError("grid has too few samples")

    actual_grid_rate = sample_count / duration
    t_grid = common_start + np.arange(sample_count, dtype=float) / actual_grid_rate
    return {
        "start": common_start,
        "end": end,
        "duration": duration,
        "cycles": selected_cycles,
        "sample_count": sample_count,
        "grid_rate_hz": actual_grid_rate,
        "fundamental_bin": selected_cycles,
        "t_grid": t_grid,
    }


def transformed_output_spectrum(output_source, y_grid, grid_rate_hz):
    y_zero = y_grid - np.mean(y_grid)
    spectrum = np.fft.rfft(y_zero)
    if output_source != "odom_position_x":
        return spectrum, y_zero

    freqs = np.fft.rfftfreq(y_zero.size, d=1.0 / grid_rate_hz)
    velocity_spectrum = spectrum * (1j * 2.0 * math.pi * freqs)
    velocity_spectrum[0] = 0.0
    velocity_grid = np.fft.irfft(velocity_spectrum, n=y_zero.size).real
    velocity_grid -= np.mean(velocity_grid)
    return velocity_spectrum, velocity_grid


def estimate_noise_floor(amplitude_spectrum, excluded_bins):
    if amplitude_spectrum.size <= 2:
        return math.nan
    mask = np.ones(amplitude_spectrum.size, dtype=bool)
    mask[0] = False
    for bin_index in excluded_bins:
        for nearby in (bin_index - 1, bin_index, bin_index + 1):
            if 0 <= nearby < mask.size:
                mask[nearby] = False
    finite = amplitude_spectrum[mask & np.isfinite(amplitude_spectrum)]
    if finite.size == 0:
        return math.nan
    return float(np.median(finite))


def max_valid_harmonic(frequency_hz, max_harmonic, nyquist_limit_hz):
    if frequency_hz <= 0.0 or not math.isfinite(nyquist_limit_hz):
        return 0
    return max(
        0,
        min(max_harmonic, int(math.floor(nyquist_limit_hz / frequency_hz + 1e-12))),
    )


def build_invalid_rows(args, experiment_dir, run_dir, run_metadata, reason):
    frequency_hz = safe_float(run_metadata.get("frequency_hz"))
    omega = 2.0 * math.pi * frequency_hz if math.isfinite(frequency_hz) else math.nan
    summary = {field: math.nan for field in FFT_SUMMARY_FIELDS}
    summary.update({
        "run_id": run_dir.name,
        "source_experiment_dir": str(experiment_dir),
        "source_run_dir": str(run_dir),
        "environment": args.environment,
        "axis": run_metadata.get("axis", "x"),
        "f_hz": frequency_hz,
        "omega_rad_s": omega,
        "valid": False,
        "reject_reason": reason,
    })
    sampling = {field: math.nan for field in SAMPLING_FIELDS}
    sampling.update({
        "run_id": run_dir.name,
        "source_run_dir": str(run_dir),
        "f_hz": frequency_hz,
        "valid": False,
        "reject_reason": reason,
    })
    return summary, [], sampling, None


def analyze_run(args, run_dir):
    source_experiment_dir = source_experiment_dir_for_run(run_dir)
    metadata_path = run_dir / "run_metadata.json"
    run_metadata = read_json(metadata_path) if metadata_path.exists() else {}

    try:
        cmd_rows = read_csv_rows(run_dir / "cmd_vel.csv")
        odom_rows = read_csv_rows(run_dir / "odom.csv")
    except FileNotFoundError as exc:
        return build_invalid_rows(args, source_experiment_dir, run_dir, run_metadata, str(exc))

    if not cmd_rows or not odom_rows:
        return build_invalid_rows(args, source_experiment_dir, run_dir, run_metadata, "missing cmd or odom samples")

    frequency_hz = safe_float(run_metadata.get("frequency_hz"), safe_float(cmd_rows[0].get("frequency_hz")))
    omega = 2.0 * math.pi * frequency_hz if math.isfinite(frequency_hz) else math.nan
    discard_cycles = safe_float(run_metadata.get("discard_cycles"), args.discard_cycles)
    ramp_cycles = safe_float(run_metadata.get("ramp_cycles"), args.ramp_cycles)

    t_cmd_raw = to_float_array(cmd_rows, "t_experiment")
    cmd_values_raw = to_float_array(cmd_rows, "cmd_vx")
    output_source, output_values_raw = select_output_source(args, odom_rows)
    t_out_raw = to_float_array(odom_rows, "t_experiment")

    cmd_stats = sampling_stats(t_cmd_raw, cmd_values_raw)
    odom_stats = sampling_stats(t_out_raw, output_values_raw)

    t_cmd, cmd_values = clean_timeseries(t_cmd_raw, cmd_values_raw)
    t_out, output_values = clean_timeseries(t_out_raw, output_values_raw)

    reject_reasons = []
    if run_metadata.get("stop_reason", "completed") != "completed":
        reject_reasons.append(f"stop_reason={run_metadata.get('stop_reason')}")
    if run_metadata.get("valid", True) is False:
        reject_reasons.append("run_metadata_valid=false")

    try:
        window = choose_integer_cycle_window(args, frequency_hz, t_cmd, t_out, discard_cycles, ramp_cycles)
    except ValueError as exc:
        reject_reasons.append(str(exc))
        summary, harmonic_rows, sampling, spectrum = build_invalid_rows(
            args, source_experiment_dir, run_dir, run_metadata, ";".join(reject_reasons)
        )
        sampling.update(prefixed_stats("cmd", cmd_stats))
        sampling.update(prefixed_stats("odom", odom_stats))
        return summary, harmonic_rows, sampling, spectrum

    t_grid = window["t_grid"]
    cmd_grid = np.interp(t_grid, t_cmd, cmd_values)
    output_grid = np.interp(t_grid, t_out, output_values)

    cmd_zero = cmd_grid - np.mean(cmd_grid)
    input_spectrum = np.fft.rfft(cmd_zero)
    output_spectrum, output_velocity_grid = transformed_output_spectrum(
        output_source, output_grid, window["grid_rate_hz"]
    )

    n = int(window["sample_count"])
    k1 = int(window["fundamental_bin"])
    spectrum_freqs = np.fft.rfftfreq(n, d=1.0 / window["grid_rate_hz"])

    input_amplitudes = 2.0 * np.abs(input_spectrum) / n
    output_amplitudes = 2.0 * np.abs(output_spectrum) / n
    excluded_bins = [h * k1 for h in range(1, args.max_harmonic + 1) if h * k1 < output_amplitudes.size]
    output_noise_floor_amp = estimate_noise_floor(output_amplitudes, excluded_bins)

    if k1 >= input_spectrum.size or k1 >= output_spectrum.size:
        reject_reasons.append("fundamental_bin_out_of_range")
        fft_gain = math.nan
        fft_phase_deg = math.nan
        fft_input_amp = math.nan
        fft_output_amp = math.nan
        fft_input_phase_deg = math.nan
        fft_output_phase_deg = math.nan
        fundamental_energy_fraction = math.nan
    else:
        fft_input_amp = float(input_amplitudes[k1])
        fft_output_amp = float(output_amplitudes[k1])
        fft_input_phase_deg = math.degrees(wrap_phase_rad(float(np.angle(input_spectrum[k1]))))
        fft_output_phase_deg = math.degrees(wrap_phase_rad(float(np.angle(output_spectrum[k1]))))
        if abs(input_spectrum[k1]) <= 0.0:
            reject_reasons.append("zero_fft_input_fundamental")
            fft_gain = math.nan
            fft_phase_deg = math.nan
        else:
            transfer = output_spectrum[k1] / input_spectrum[k1]
            fft_gain = float(abs(transfer))
            fft_phase_deg = math.degrees(wrap_phase_rad(float(np.angle(transfer))))

        total_output_power = float(np.mean(output_velocity_grid * output_velocity_grid))
        fundamental_energy_fraction = (
            (fft_output_amp * fft_output_amp / 2.0) / total_output_power
            if total_output_power > 0.0 and math.isfinite(fft_output_amp)
            else math.nan
        )

    raw_rates = [
        cmd_stats.get("median_rate_hz", math.nan),
        odom_stats.get("median_rate_hz", math.nan),
        window["grid_rate_hz"],
    ]
    finite_rates = [rate for rate in raw_rates if math.isfinite(rate) and rate > 0.0]
    raw_min_nyquist_hz = 0.5 * min(finite_rates) if finite_rates else math.nan
    nyquist_limit_hz = args.nyquist_safety * raw_min_nyquist_hz if math.isfinite(raw_min_nyquist_hz) else math.nan
    max_valid_h = max_valid_harmonic(frequency_hz, args.max_harmonic, nyquist_limit_hz)

    harmonic_rows = []
    output_harmonic_amps_for_thd = []
    odd_output_harmonic_amps_for_thd = []
    for harmonic in range(1, args.max_harmonic + 1):
        bin_index = harmonic * k1
        harmonic_frequency_hz = harmonic * frequency_hz
        nyquist_valid = (
            bin_index < input_amplitudes.size
            and bin_index < output_amplitudes.size
            and harmonic <= max_valid_h
        )
        input_amp = float(input_amplitudes[bin_index]) if bin_index < input_amplitudes.size else math.nan
        output_amp = float(output_amplitudes[bin_index]) if bin_index < output_amplitudes.size else math.nan
        input_relative = input_amp / fft_input_amp if fft_input_amp > 0.0 and math.isfinite(input_amp) else math.nan
        output_relative = output_amp / fft_output_amp if fft_output_amp > 0.0 and math.isfinite(output_amp) else math.nan
        total_output_power = float(np.mean(output_velocity_grid * output_velocity_grid))
        output_energy_fraction = (
            (output_amp * output_amp / 2.0) / total_output_power
            if total_output_power > 0.0 and math.isfinite(output_amp)
            else math.nan
        )
        output_snr_db = (
            20.0 * math.log10(output_amp / output_noise_floor_amp)
            if output_noise_floor_amp > 0.0 and output_amp > 0.0
            else math.nan
        )
        if harmonic >= 2 and nyquist_valid and math.isfinite(output_amp):
            output_harmonic_amps_for_thd.append(output_amp)
            if harmonic in (3, 5, 7):
                odd_output_harmonic_amps_for_thd.append(output_amp)
        harmonic_rows.append({
            "run_id": run_dir.name,
            "source_run_dir": str(run_dir),
            "f_hz": frequency_hz,
            "harmonic": harmonic,
            "harmonic_frequency_hz": harmonic_frequency_hz,
            "bin_index": bin_index,
            "nyquist_valid": nyquist_valid,
            "input_amp": input_amp,
            "output_amp": output_amp,
            "input_amp_relative_to_fundamental": input_relative,
            "output_amp_relative_to_fundamental": output_relative,
            "output_energy_fraction": output_energy_fraction,
            "output_noise_floor_amp": output_noise_floor_amp,
            "output_snr_db": output_snr_db,
            "valid": len(reject_reasons) == 0,
            "reject_reason": ";".join(reject_reasons),
        })

    thd = (
        math.sqrt(sum(value * value for value in output_harmonic_amps_for_thd)) / fft_output_amp
        if fft_output_amp > 0.0 and output_harmonic_amps_for_thd
        else math.nan
    )
    odd_thd = (
        math.sqrt(sum(value * value for value in odd_output_harmonic_amps_for_thd)) / fft_output_amp
        if fft_output_amp > 0.0 and odd_output_harmonic_amps_for_thd
        else math.nan
    )
    for row in harmonic_rows:
        row["thd_2_to_max"] = thd
        row["odd_thd_3_5_7"] = odd_thd

    summary = {
        "run_id": run_dir.name,
        "source_experiment_dir": str(source_experiment_dir),
        "source_run_dir": str(run_dir),
        "environment": args.environment,
        "axis": run_metadata.get("axis", "x"),
        "output_source": output_source,
        "f_hz": frequency_hz,
        "omega_rad_s": omega,
        "A_cmd_nominal": safe_float(run_metadata.get("amplitude")),
        "analysis_start_s": window["start"],
        "analysis_end_s": window["end"],
        "analysis_duration_s": window["duration"],
        "analysis_cycles": window["cycles"],
        "grid_sample_count": window["sample_count"],
        "grid_rate_hz": window["grid_rate_hz"],
        "fundamental_bin": k1,
        "fft_gain_linear": fft_gain,
        "fft_gain_db": gain_db(fft_gain),
        "fft_phase_deg": fft_phase_deg,
        "fft_input_amp": fft_input_amp,
        "fft_output_amp": fft_output_amp,
        "fft_input_phase_deg": fft_input_phase_deg,
        "fft_output_phase_deg": fft_output_phase_deg,
        "fft_output_fundamental_energy_fraction": fundamental_energy_fraction,
        "fft_output_noise_floor_amp": output_noise_floor_amp,
        "thd_2_to_max": thd,
        "odd_thd_3_5_7": odd_thd,
        "valid": len(reject_reasons) == 0,
        "reject_reason": ";".join(reject_reasons),
    }

    sampling = {
        "run_id": run_dir.name,
        "source_run_dir": str(run_dir),
        "f_hz": frequency_hz,
        "raw_min_nyquist_hz": raw_min_nyquist_hz,
        "grid_rate_hz": window["grid_rate_hz"],
        "grid_nyquist_hz": 0.5 * window["grid_rate_hz"],
        "nyquist_safety_factor": args.nyquist_safety,
        "max_valid_harmonic_by_nyquist": max_valid_h,
        "analysis_start_s": window["start"],
        "analysis_end_s": window["end"],
        "analysis_duration_s": window["duration"],
        "analysis_cycles": window["cycles"],
        "grid_sample_count": window["sample_count"],
        "valid": len(reject_reasons) == 0,
        "reject_reason": ";".join(reject_reasons),
    }
    sampling.update(prefixed_stats("cmd", cmd_stats))
    sampling.update(prefixed_stats("odom", odom_stats))

    spectrum = {
        "run_id": run_dir.name,
        "f_hz": frequency_hz,
        "freqs": spectrum_freqs,
        "input_amplitudes": input_amplitudes,
        "output_amplitudes": output_amplitudes,
        "nyquist_limit_hz": nyquist_limit_hz,
        "valid": len(reject_reasons) == 0,
    }
    return summary, harmonic_rows, sampling, spectrum


def prefixed_stats(prefix, stats):
    return {
        f"{prefix}_raw_sample_count": stats["raw_sample_count"],
        f"{prefix}_finite_pair_count": stats["finite_pair_count"],
        f"{prefix}_nonmonotonic_count": stats["nonmonotonic_count"],
        f"{prefix}_duplicate_time_count": stats["duplicate_time_count"],
        f"{prefix}_t_start_s": stats["t_start_s"],
        f"{prefix}_t_end_s": stats["t_end_s"],
        f"{prefix}_duration_s": stats["duration_s"],
        f"{prefix}_median_dt_s": stats["median_dt_s"],
        f"{prefix}_min_dt_s": stats["min_dt_s"],
        f"{prefix}_max_dt_s": stats["max_dt_s"],
        f"{prefix}_dt_cv": stats["dt_cv"],
        f"{prefix}_median_rate_hz": stats["median_rate_hz"],
    }


def write_plots(analysis_dir, summary_rows, harmonic_rows, spectra, args):
    valid_rows = [row for row in summary_rows if row.get("valid") is True]
    valid_rows.sort(key=lambda row: safe_float(row.get("f_hz")))

    if valid_rows:
        freqs = np.asarray([row["f_hz"] for row in valid_rows], dtype=float)
        fig, ax = plt.subplots(figsize=(7, 4))
        values = np.asarray([safe_float(row.get("fft_gain_db")) for row in valid_rows], dtype=float)
        mask = np.isfinite(values)
        if np.any(mask):
            ax.plot(freqs[mask], values[mask], marker="o", label="FFT fundamental")
        ax.set_xscale("log")
        ax.set_xlabel("Excitation frequency (Hz)")
        ax.set_ylabel("Gain (dB)")
        ax.grid(True, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(analysis_dir / "gain_fft.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        values = np.asarray([safe_float(row.get("fft_phase_deg")) for row in valid_rows], dtype=float)
        mask = np.isfinite(values)
        if np.any(mask):
            ax.plot(freqs[mask], values[mask], marker="o", label="FFT fundamental")
        ax.set_xscale("log")
        ax.set_xlabel("Excitation frequency (Hz)")
        ax.set_ylabel("Phase (deg)")
        ax.grid(True, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(analysis_dir / "phase_fft.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for harmonic in range(2, args.max_harmonic + 1):
        rows = [
            row for row in harmonic_rows
            if int(row.get("harmonic", -1)) == harmonic and row.get("nyquist_valid") is True
        ]
        rows.sort(key=lambda row: safe_float(row.get("f_hz")))
        if not rows:
            continue
        x = np.asarray([row["f_hz"] for row in rows], dtype=float)
        y = np.asarray([100.0 * safe_float(row.get("output_amp_relative_to_fundamental")) for row in rows], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.any(mask):
            ax.plot(x[mask], y[mask], marker="o", label=f"H{harmonic}")
            plotted = True
    if plotted:
        ax.set_xscale("log")
        ax.set_xlabel("Excitation frequency (Hz)")
        ax.set_ylabel("Output harmonic / fundamental (%)")
        ax.grid(True, which="both")
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(analysis_dir / "harmonic_magnitudes.png", dpi=160)
    plt.close(fig)

    spectra_to_plot = [spectrum for spectrum in spectra if spectrum and spectrum.get("valid")]
    spectra_to_plot.sort(key=lambda item: safe_float(item.get("f_hz")))
    for spectrum in spectra_to_plot[: max(0, args.max_spectrum_plots)]:
        freqs = spectrum["freqs"]
        input_amplitudes = spectrum["input_amplitudes"]
        output_amplitudes = spectrum["output_amplitudes"]
        max_freq = spectrum.get("nyquist_limit_hz", math.nan)
        if not math.isfinite(max_freq) or max_freq <= 0.0:
            max_freq = float(freqs[-1])
        mask = (freqs > 0.0) & (freqs <= max_freq)
        if not np.any(mask):
            continue
        floor = 1e-12
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(freqs[mask], np.maximum(input_amplitudes[mask], floor), marker=".", label="cmd_vx")
        ax.semilogy(freqs[mask], np.maximum(output_amplitudes[mask], floor), marker=".", label="output velocity")
        ax.axvline(spectrum["f_hz"], color="black", linestyle="--", linewidth=1.0, label="excitation")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Single-sided amplitude")
        ax.grid(True, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(analysis_dir / f"spectra_{spectrum['run_id']}.png", dpi=160)
        plt.close(fig)


def json_ready(value):
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def spectra_for_html(spectra, max_spectrum_plots):
    selected = [spectrum for spectrum in spectra if spectrum and spectrum.get("valid")]
    selected.sort(key=lambda item: safe_float(item.get("f_hz")))
    html_spectra = []
    for spectrum in selected[: max(0, max_spectrum_plots)]:
        freqs = spectrum["freqs"]
        input_amplitudes = spectrum["input_amplitudes"]
        output_amplitudes = spectrum["output_amplitudes"]
        max_freq = spectrum.get("nyquist_limit_hz", math.nan)
        mask = freqs > 0.0
        if math.isfinite(max_freq) and max_freq > 0.0:
            mask = mask & (freqs <= max_freq)
        html_spectra.append({
            "run_id": spectrum["run_id"],
            "f_hz": spectrum["f_hz"],
            "nyquist_limit_hz": max_freq,
            "freqs": freqs[mask],
            "input_amplitudes": input_amplitudes[mask],
            "output_amplitudes": output_amplitudes[mask],
        })
    return html_spectra


def write_html_report(analysis_dir, html_path, summary_rows, harmonic_rows, sampling_rows, spectra, args, experiment_dir):
    valid_count = sum(1 for row in summary_rows if row.get("valid") is True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_dir": str(experiment_dir),
        "analysis_dir": str(analysis_dir),
        "run_count": len(summary_rows),
        "valid_count": valid_count,
        "parameters": {
            "environment": args.environment,
            "output_source": args.output_source,
            "analysis_cycles": args.analysis_cycles,
            "min_analysis_cycles": args.min_analysis_cycles,
            "grid_rate_hz": args.grid_rate_hz,
            "max_harmonic": args.max_harmonic,
            "nyquist_safety": args.nyquist_safety,
        },
        "fft_rows": summary_rows,
        "harmonic_rows": harmonic_rows,
        "sampling_rows": sampling_rows,
        "spectra": spectra_for_html(spectra, args.max_spectrum_plots),
    }
    data_json = json.dumps(json_ready(payload), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RB-KAIROS FFT 频率响应报告</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #6b7280;
      --grid: #e5e7eb;
      --accent: #2563eb;
      --good: #047857;
      --warn: #b45309;
      --bad: #b91c1c;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.45;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 18px 48px;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 21px; margin-top: 28px; }}
    h3 {{ font-size: 17px; margin-top: 18px; }}
    p {{ margin: 8px 0; }}
    code {{
      background: #eef2ff;
      padding: 1px 5px;
      border-radius: 4px;
      font-size: 0.92em;
    }}
    .muted {{ color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
      margin: 14px 0 18px;
    }}
    .card, .panel {{
      background: var(--panel);
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }}
    .card {{ padding: 14px 16px; }}
    .card .label {{ color: var(--muted); font-size: 13px; }}
    .card .value {{ font-size: 24px; font-weight: 700; margin-top: 5px; }}
    .panel {{ padding: 16px; margin-top: 14px; overflow-x: auto; }}
    .chart svg {{ width: 100%; height: 340px; display: block; }}
    .spectra-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 14px;
    }}
    .spectra-grid .chart svg {{ height: 280px; }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      margin: 8px 0 0;
      font-size: 13px;
      color: var(--muted);
    }}
    .legend span::before {{
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 6px;
      background: var(--c);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      white-space: nowrap;
    }}
    th, td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 7px 8px;
      text-align: right;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{
      background: #f9fafb;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    .heatmap {{
      display: grid;
      grid-template-columns: minmax(90px, 120px) repeat(var(--cols), minmax(70px, 1fr));
      gap: 3px;
      min-width: 640px;
      font-size: 12px;
    }}
    .heatmap .cell {{
      padding: 8px;
      border-radius: 6px;
      text-align: center;
      background: #f3f4f6;
    }}
    .heatmap .head {{ font-weight: 700; background: #e5e7eb; }}
    .status-good {{ color: var(--good); }}
    .status-warn {{ color: var(--warn); }}
  </style>
</head>
<body>
<main>
  <h1>RB-KAIROS FFT 频率响应报告</h1>
  <p class="muted">自包含 HTML；图形由浏览器使用内嵌 JSON 和原生 SVG 绘制，不依赖 Matplotlib、外部 JS 或 CDN。</p>

  <section class="grid" id="summaryCards"></section>

  <section class="panel">
    <h2>FFT 基频频率响应</h2>
    <p class="muted">增益和相位直接来自重采样整数周期窗口的 FFT 基频 bin。</p>
    <div class="chart" id="gainChart"></div>
    <div class="legend">
      <span style="--c:#dc2626">FFT fundamental</span>
    </div>
    <div class="chart" id="phaseChart"></div>
  </section>

  <section class="panel">
    <h2>谐波和能量诊断</h2>
    <div class="chart" id="fundamentalEnergyChart"></div>
    <div class="chart" id="thdChart"></div>
    <h3>输出谐波 / 基频幅值比例</h3>
    <div id="harmonicHeatmap"></div>
  </section>

  <section class="panel">
    <h2>示例频谱</h2>
    <p class="muted">幅值为单边谱幅值。显示范围受原始采样率和 Nyquist 安全系数限制。</p>
    <div class="spectra-grid" id="spectraCharts"></div>
  </section>

  <section class="panel">
    <h2>详细结果表</h2>
    <div id="fftTable"></div>
  </section>

  <section class="panel">
    <h2>采样诊断表</h2>
    <div id="samplingTable"></div>
  </section>
</main>

<script>
const DATA = {data_json};

function finiteNumber(value) {{
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}}

function fmt(value, digits = 4) {{
  const n = finiteNumber(value);
  if (n === null) return "—";
  if (Math.abs(n) >= 1000 || (Math.abs(n) > 0 && Math.abs(n) < 0.001)) return n.toExponential(3);
  return n.toFixed(digits).replace(/\\.0+$/, "").replace(/(\\.\\d*?)0+$/, "$1");
}}

function pct(value, digits = 1) {{
  const n = finiteNumber(value);
  return n === null ? "—" : fmt(100 * n, digits) + "%";
}}

function seriesFromRows(rows, field, label, color) {{
  return {{
    label,
    color,
    points: rows.map(r => [finiteNumber(r.f_hz), finiteNumber(r[field])]).filter(p => p[0] !== null && p[1] !== null)
  }};
}}

function extent(values, padFrac = 0.08) {{
  const clean = values.filter(Number.isFinite);
  if (!clean.length) return [0, 1];
  let lo = Math.min(...clean);
  let hi = Math.max(...clean);
  if (lo === hi) {{
    const pad = Math.abs(lo || 1) * 0.1;
    lo -= pad;
    hi += pad;
  }}
  const pad = (hi - lo) * padFrac;
  return [lo - pad, hi + pad];
}}

function makeSvgChart(targetId, title, xLabel, yLabel, series, opts = {{}}) {{
  const target = document.getElementById(targetId);
  const width = 980;
  const height = opts.height || 330;
  const margin = {{left: 64, right: 22, top: 34, bottom: 54}};
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const allPoints = series.flatMap(s => s.points);
  target.innerHTML = "";
  if (!allPoints.length) {{
    target.textContent = "无可绘制数据";
    return;
  }}
  const xValsRaw = allPoints.map(p => p[0]).filter(v => Number.isFinite(v) && v > 0);
  const yValsRaw = allPoints.map(p => p[1]).filter(v => Number.isFinite(v));
  const useLogX = opts.logX !== false;
  const useLogY = opts.logY === true;
  const xVals = useLogX ? xValsRaw.map(Math.log10) : xValsRaw;
  const yVals = useLogY ? yValsRaw.filter(v => v > 0).map(Math.log10) : yValsRaw;
  const [xMin, xMax] = extent(xVals, 0.04);
  const [yMin, yMax] = opts.yDomain || extent(yVals, 0.10);
  const sx = x => margin.left + ((useLogX ? Math.log10(x) : x) - xMin) / (xMax - xMin) * plotW;
  const sy = y => margin.top + plotH - ((useLogY ? Math.log10(Math.max(y, 1e-15)) : y) - yMin) / (yMax - yMin) * plotH;

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);

  function line(x1, y1, x2, y2, stroke = "#e5e7eb", width = 1) {{
    const el = document.createElementNS(svg.namespaceURI, "line");
    el.setAttribute("x1", x1); el.setAttribute("y1", y1);
    el.setAttribute("x2", x2); el.setAttribute("y2", y2);
    el.setAttribute("stroke", stroke); el.setAttribute("stroke-width", width);
    svg.appendChild(el);
  }}
  function text(x, y, value, anchor = "middle", size = 12, color = "#6b7280") {{
    const el = document.createElementNS(svg.namespaceURI, "text");
    el.setAttribute("x", x); el.setAttribute("y", y);
    el.setAttribute("text-anchor", anchor); el.setAttribute("font-size", size);
    el.setAttribute("fill", color); el.textContent = value;
    svg.appendChild(el);
  }}

  text(width / 2, 20, title, "middle", 15, "#1f2937");
  line(margin.left, margin.top, margin.left, margin.top + plotH, "#9ca3af", 1.2);
  line(margin.left, margin.top + plotH, margin.left + plotW, margin.top + plotH, "#9ca3af", 1.2);

  const xTicks = Array.from(new Set(xValsRaw.map(v => Number(v.toPrecision(6))))).sort((a, b) => a - b);
  xTicks.forEach(v => {{
    const x = sx(v);
    if (x < margin.left - 1 || x > margin.left + plotW + 1) return;
    line(x, margin.top, x, margin.top + plotH, "#eef2f7");
    text(x, margin.top + plotH + 18, fmt(v, 3), "middle", 11);
  }});

  for (let i = 0; i <= 5; i++) {{
    const yy = yMin + (yMax - yMin) * i / 5;
    const y = margin.top + plotH - plotH * i / 5;
    line(margin.left, y, margin.left + plotW, y, "#eef2f7");
    text(margin.left - 9, y + 4, useLogY ? fmt(Math.pow(10, yy), 3) : fmt(yy, 3), "end", 11);
  }}
  text(width / 2, height - 12, xLabel, "middle", 12);
  const yLabelEl = document.createElementNS(svg.namespaceURI, "text");
  yLabelEl.setAttribute("x", 16); yLabelEl.setAttribute("y", height / 2);
  yLabelEl.setAttribute("text-anchor", "middle"); yLabelEl.setAttribute("font-size", "12");
  yLabelEl.setAttribute("fill", "#6b7280");
  yLabelEl.setAttribute("transform", `rotate(-90 16 ${{height / 2}})`);
  yLabelEl.textContent = yLabel;
  svg.appendChild(yLabelEl);

  series.forEach(s => {{
    const pts = s.points.filter(p => p[0] !== null && p[1] !== null && (!useLogY || p[1] > 0));
    if (!pts.length) return;
    const path = document.createElementNS(svg.namespaceURI, "path");
    path.setAttribute("d", pts.map((p, i) => `${{i ? "L" : "M"}} ${{sx(p[0])}} ${{sy(p[1])}}`).join(" "));
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", s.color);
    path.setAttribute("stroke-width", "2");
    svg.appendChild(path);
    pts.forEach(p => {{
      const c = document.createElementNS(svg.namespaceURI, "circle");
      c.setAttribute("cx", sx(p[0])); c.setAttribute("cy", sy(p[1]));
      c.setAttribute("r", "3.4"); c.setAttribute("fill", s.color);
      const titleNode = document.createElementNS(svg.namespaceURI, "title");
      titleNode.textContent = `${{s.label}}\\nf=${{fmt(p[0], 3)}} Hz\\ny=${{fmt(p[1], 5)}}`;
      c.appendChild(titleNode);
      svg.appendChild(c);
    }});
  }});
  target.appendChild(svg);
}}

function renderCards() {{
  const cards = [
    ["输入目录", DATA.experiment_dir],
    ["有效 run", `${{DATA.valid_count}} / ${{DATA.run_count}}`],
    ["网格率", `${{fmt(DATA.parameters.grid_rate_hz, 2)}} Hz`],
    ["最大谐波", `H${{DATA.parameters.max_harmonic}}`],
    ["生成时间 UTC", DATA.generated_at_utc],
  ];
  const container = document.getElementById("summaryCards");
  cards.forEach(([label, value]) => {{
    const div = document.createElement("div");
    div.className = "card";
    div.innerHTML = `<div class="label"></div><div class="value"></div>`;
    div.children[0].textContent = label;
    div.children[1].textContent = value;
    if (String(value).length > 32) div.children[1].style.fontSize = "14px";
    container.appendChild(div);
  }});
}}

function renderHeatmap() {{
  const rows = DATA.fft_rows.filter(r => r.valid);
  const harmonics = [];
  for (let h = 2; h <= DATA.parameters.max_harmonic; h++) harmonics.push(h);
  const root = document.getElementById("harmonicHeatmap");
  root.style.setProperty("--cols", harmonics.length);
  const map = new Map();
  DATA.harmonic_rows.forEach(r => map.set(`${{r.run_id}}:${{r.harmonic}}`, r));
  const html = [];
  html.push(`<div class="heatmap" style="--cols:${{harmonics.length}}">`);
  html.push(`<div class="cell head">run</div>`);
  harmonics.forEach(h => html.push(`<div class="cell head">H${{h}}</div>`));
  rows.forEach(row => {{
    html.push(`<div class="cell head">${{row.run_id}}</div>`);
    harmonics.forEach(h => {{
      const hr = map.get(`${{row.run_id}}:${{h}}`);
      const val = hr && hr.nyquist_valid ? finiteNumber(hr.output_amp_relative_to_fundamental) : null;
      const alpha = val === null ? 0 : Math.min(1, Math.max(0, val / 0.35));
      const bg = val === null ? "#f3f4f6" : `rgba(220, 38, 38, ${{0.12 + 0.70 * alpha}})`;
      const text = val === null ? "N/A" : pct(val, 1);
      html.push(`<div class="cell" style="background:${{bg}}">${{text}}</div>`);
    }});
  }});
  html.push(`</div>`);
  root.innerHTML = html.join("");
}}

function renderTable(targetId, rows, columns) {{
  const root = document.getElementById(targetId);
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const trh = document.createElement("tr");
  columns.forEach(([field, label]) => {{
    const th = document.createElement("th");
    th.textContent = label;
    trh.appendChild(th);
  }});
  thead.appendChild(trh);
  table.appendChild(thead);
  const tbody = document.createElement("tbody");
  rows.forEach(row => {{
    const tr = document.createElement("tr");
    columns.forEach(([field]) => {{
      const td = document.createElement("td");
      const value = row[field];
      if (field.includes("phase")) td.textContent = fmt(value, 2);
      else if (field.includes("gain") || field.includes("thd") || field.includes("fraction")) td.textContent = fmt(value, 4);
      else if (field === "valid") td.textContent = value ? "true" : "false";
      else td.textContent = typeof value === "number" ? fmt(value, 4) : (value ?? "—");
      tr.appendChild(td);
    }});
    tbody.appendChild(tr);
  }});
  table.appendChild(tbody);
  root.appendChild(table);
}}

function renderSpectra() {{
  const root = document.getElementById("spectraCharts");
  DATA.spectra.forEach((s, idx) => {{
    const div = document.createElement("div");
    div.className = "panel chart";
    const id = `spectrum_${{idx}}`;
    div.innerHTML = `<h3>${{s.run_id}} (${{fmt(s.f_hz, 3)}} Hz)</h3><div id="${{id}}"></div>`;
    root.appendChild(div);
    const input = s.freqs.map((f, i) => [finiteNumber(f), finiteNumber(s.input_amplitudes[i])]).filter(p => p[0] !== null && p[1] !== null && p[1] > 0);
    const output = s.freqs.map((f, i) => [finiteNumber(f), finiteNumber(s.output_amplitudes[i])]).filter(p => p[0] !== null && p[1] !== null && p[1] > 0);
    makeSvgChart(id, "单边幅值谱", "Frequency (Hz)", "Amplitude", [
      {{label: "cmd_vx", color: "#2563eb", points: input}},
      {{label: "output", color: "#dc2626", points: output}},
    ], {{logX: false, logY: true, height: 260}});
  }});
}}

function main() {{
  const rows = DATA.fft_rows.filter(r => r.valid).sort((a, b) => a.f_hz - b.f_hz);
  renderCards();
  makeSvgChart("gainChart", "FFT 基频增益", "Excitation frequency (Hz)", "Gain (dB)", [
    seriesFromRows(rows, "fft_gain_db", "FFT fundamental", "#dc2626"),
  ]);
  makeSvgChart("phaseChart", "FFT 基频相位", "Excitation frequency (Hz)", "Phase (deg)", [
    seriesFromRows(rows, "fft_phase_deg", "FFT fundamental", "#dc2626"),
  ]);
  makeSvgChart("fundamentalEnergyChart", "输出基频能量占比", "Excitation frequency (Hz)", "Energy fraction", [
    seriesFromRows(rows, "fft_output_fundamental_energy_fraction", "fundamental energy", "#7c3aed"),
  ], {{yDomain: [0, 1.05]}});
  makeSvgChart("thdChart", "输出 THD（有效谐波）", "Excitation frequency (Hz)", "THD", [
    seriesFromRows(rows, "thd_2_to_max", "THD H2-Hmax", "#ea580c"),
    seriesFromRows(rows, "odd_thd_3_5_7", "odd THD H3/H5/H7", "#0891b2"),
  ]);
  renderHeatmap();
  renderSpectra();
  renderTable("fftTable", DATA.fft_rows, [
    ["run_id", "run"],
    ["f_hz", "f Hz"],
    ["analysis_cycles", "cycles"],
    ["fft_gain_db", "FFT gain dB"],
    ["fft_phase_deg", "FFT phase deg"],
    ["fft_input_amp", "FFT input amp"],
    ["fft_output_amp", "FFT output amp"],
    ["fft_input_phase_deg", "input phase deg"],
    ["fft_output_phase_deg", "output phase deg"],
    ["fft_output_fundamental_energy_fraction", "fund energy"],
    ["thd_2_to_max", "THD"],
    ["valid", "valid"],
    ["reject_reason", "reject reason"],
  ]);
  renderTable("samplingTable", DATA.sampling_rows, [
    ["run_id", "run"],
    ["f_hz", "f Hz"],
    ["cmd_median_rate_hz", "cmd Hz"],
    ["odom_median_rate_hz", "odom Hz"],
    ["cmd_dt_cv", "cmd dt CV"],
    ["odom_dt_cv", "odom dt CV"],
    ["grid_rate_hz", "grid Hz"],
    ["max_valid_harmonic_by_nyquist", "max valid H"],
    ["analysis_cycles", "cycles"],
    ["valid", "valid"],
  ]);
}}

main();
</script>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_text, encoding="utf-8")


def write_metadata(analysis_dir, args, experiment_dir, run_count, valid_count, plot_status=None):
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "analyze_frequency_response_fft.py",
        "experiment_dir": str(experiment_dir),
        "run_count": run_count,
        "valid_count": valid_count,
        "outputs": {
            "frequency_fft_summary": "frequency_fft_summary.csv",
            "harmonic_summary": "harmonic_summary.csv",
            "sampling_diagnostics": "sampling_diagnostics.csv",
            "html_report": (args.html_output or "frequency_response_fft_report.html") if not args.skip_html else None,
            "plots": plot_status,
        },
        "assumptions": [
            "input signal is cmd_vel.csv:cmd_vx",
            "default output signal is odom.csv:vx unless --output-source selects position fallback",
            "FFT is computed only after interpolation to a uniform grid",
            "analysis window starts after both discard and ramp intervals",
            "analysis window contains an integer number of excitation periods",
            "signals are mean-removed before FFT; no linear detrending is applied",
            "harmonic validity is constrained by raw command/output sampling rates and grid Nyquist frequency",
        ],
        "parameters": {
            "environment": args.environment,
            "output_source": args.output_source,
            "analysis_cycles": args.analysis_cycles,
            "cycle_shortfall_tolerance": args.cycle_shortfall_tolerance,
            "min_analysis_cycles": args.min_analysis_cycles,
            "grid_rate_hz_requested": args.grid_rate_hz,
            "discard_cycles_default": args.discard_cycles,
            "ramp_cycles_default": args.ramp_cycles,
            "max_harmonic": args.max_harmonic,
            "nyquist_safety": args.nyquist_safety,
            "plots_requested": not args.skip_plots,
            "html_requested": not args.skip_html,
            "html_output": args.html_output,
        },
    }
    with (analysis_dir / "analysis_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def analyze_frequency_response_fft(
    experiment_dir,
    *,
    output_dir=None,
    environment="auto",
    output_source="auto",
    discard_cycles=2.0,
    ramp_cycles=1.0,
    analysis_cycles=4,
    min_analysis_cycles=2.0,
    grid_rate_hz=50.0,
    cycle_shortfall_tolerance=0.05,
    max_harmonic=7,
    nyquist_safety=0.90,
    plots=True,
    html=True,
    html_output=None,
    max_spectrum_plots=8,
):
    """Analyze one frequency-response experiment with the FFT workflow."""
    experiment_dir = Path(experiment_dir)
    run_dirs = collect_run_dirs(experiment_dir)
    if not run_dirs:
        raise RuntimeError(f"No run directories found in {experiment_dir}")

    try:
        load_numpy()
    except Exception as exc:
        raise RuntimeError(f"numpy is required for FFT analysis: {exc}") from exc

    args = argparse.Namespace(
        experiment_dir=str(experiment_dir),
        output_dir=str(output_dir) if output_dir is not None else None,
        environment=str(environment),
        output_source=str(output_source),
        discard_cycles=float(discard_cycles),
        ramp_cycles=float(ramp_cycles),
        analysis_cycles=int(analysis_cycles),
        min_analysis_cycles=float(min_analysis_cycles),
        grid_rate_hz=float(grid_rate_hz),
        cycle_shortfall_tolerance=float(cycle_shortfall_tolerance),
        max_harmonic=int(max_harmonic),
        nyquist_safety=float(nyquist_safety),
        skip_plots=not bool(plots),
        skip_html=not bool(html),
        html_output=str(html_output) if html_output is not None else None,
        max_spectrum_plots=int(max_spectrum_plots),
    )

    analysis_dir = (
        Path(args.output_dir) if args.output_dir else experiment_dir / "fft_analysis"
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    harmonic_rows = []
    sampling_rows = []
    spectra = []
    for run_dir in run_dirs:
        summary, harmonics, sampling, spectrum = analyze_run(args, run_dir)
        summary_rows.append(summary)
        harmonic_rows.extend(harmonics)
        sampling_rows.append(sampling)
        if spectrum is not None:
            spectra.append(spectrum)

    summary_rows.sort(key=lambda row: safe_float(row.get("f_hz")))
    harmonic_rows.sort(
        key=lambda row: (safe_float(row.get("f_hz")), int(row.get("harmonic", 0)))
    )
    sampling_rows.sort(key=lambda row: safe_float(row.get("f_hz")))

    summary_csv = analysis_dir / "frequency_fft_summary.csv"
    harmonic_csv = analysis_dir / "harmonic_summary.csv"
    sampling_csv = analysis_dir / "sampling_diagnostics.csv"
    write_csv(summary_csv, summary_rows, FFT_SUMMARY_FIELDS)
    write_csv(harmonic_csv, harmonic_rows, HARMONIC_FIELDS)
    write_csv(sampling_csv, sampling_rows, SAMPLING_FIELDS)

    plot_status = {"generated": False, "reason": "disabled"}
    if not args.skip_plots:
        if load_matplotlib():
            write_plots(analysis_dir, summary_rows, harmonic_rows, spectra, args)
            plot_status = {"generated": True, "format": "png"}
        else:
            svg_files = write_svg_plots(analysis_dir, summary_rows, harmonic_rows, spectra, args)
            plot_status = {
                "generated": bool(svg_files),
                "format": "svg",
                "reason": "matplotlib_unavailable",
                "files": svg_files,
            }

    html_path = None
    if not args.skip_html:
        html_path = (
            Path(args.html_output)
            if args.html_output
            else analysis_dir / "frequency_response_fft_report.html"
        )
        write_html_report(
            analysis_dir,
            html_path,
            summary_rows,
            harmonic_rows,
            sampling_rows,
            spectra,
            args,
            experiment_dir,
        )

    valid_count = sum(1 for row in summary_rows if row.get("valid") is True)
    write_metadata(
        analysis_dir,
        args,
        experiment_dir,
        len(summary_rows),
        valid_count,
        plot_status=plot_status,
    )

    source_experiment_dirs = []
    seen_source_dirs = set()
    for run_dir in run_dirs:
        source_dir = source_experiment_dir_for_run(run_dir)
        key = str(source_dir)
        if key not in seen_source_dirs:
            seen_source_dirs.add(key)
            source_experiment_dirs.append(source_dir)

    return {
        "summary_csv": str(summary_csv),
        "harmonic_csv": str(harmonic_csv),
        "sampling_csv": str(sampling_csv),
        "html_path": str(html_path) if html_path is not None else None,
        "output_dir": str(analysis_dir),
        "input_dir": str(experiment_dir),
        "source_experiment_dirs": [str(path) for path in source_experiment_dirs],
        "source_run_dirs": [str(path) for path in run_dirs],
        "run_count": len(summary_rows),
        "valid_run_count": valid_count,
        "plot_status": plot_status,
        "summaries": summary_rows,
        "harmonic_rows": harmonic_rows,
        "sampling_rows": sampling_rows,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="FFT-based analysis for RBKairos frequency-response CSV output."
    )
    parser.add_argument("--experiment-dir", required=True, help="Experiment directory, run directory, or grouped directory.")
    parser.add_argument("--output-dir", help="Defaults to <experiment-dir>/fft_analysis.")
    parser.add_argument("--environment", default="auto", help="Free-form label written to output CSV files.")
    parser.add_argument(
        "--output-source",
        choices=["auto", "odom_twist_vx", "odom_position_x"],
        default="auto",
        help="Output signal to analyze. auto prefers odom.csv:vx when it varies.",
    )
    parser.add_argument("--discard-cycles", type=float, default=2.0, help="Fallback discard cycles when metadata is absent.")
    parser.add_argument("--ramp-cycles", type=float, default=1.0, help="Fallback ramp cycles when metadata is absent.")
    parser.add_argument(
        "--analysis-cycles",
        type=int,
        default=4,
        help="Requested integer steady-state cycles for the FFT window. Reduced if less data are available.",
    )
    parser.add_argument("--min-analysis-cycles", type=float, default=2.0)
    parser.add_argument("--grid-rate-hz", type=float, default=50.0)
    parser.add_argument(
        "--cycle-shortfall-tolerance",
        type=float,
        default=0.05,
        help="Allow this many missing cycles by shifting the window start earlier instead of reducing cycle count.",
    )
    parser.add_argument("--max-harmonic", type=int, default=7)
    parser.add_argument(
        "--nyquist-safety",
        type=float,
        default=0.90,
        help="Require harmonic frequency <= this factor times the minimum raw/grid Nyquist.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-html", action="store_true", help="Do not write the self-contained HTML report.")
    parser.add_argument("--html-output", help="Optional HTML report path. Defaults to <output-dir>/frequency_response_fft_report.html.")
    parser.add_argument("--max-spectrum-plots", type=int, default=8)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        result = analyze_frequency_response_fft(
            Path(args.experiment_dir),
            output_dir=Path(args.output_dir) if args.output_dir else None,
            environment=args.environment,
            output_source=args.output_source,
            discard_cycles=args.discard_cycles,
            ramp_cycles=args.ramp_cycles,
            analysis_cycles=args.analysis_cycles,
            min_analysis_cycles=args.min_analysis_cycles,
            grid_rate_hz=args.grid_rate_hz,
            cycle_shortfall_tolerance=args.cycle_shortfall_tolerance,
            max_harmonic=args.max_harmonic,
            nyquist_safety=args.nyquist_safety,
            plots=not args.skip_plots,
            html=not args.skip_html,
            html_output=Path(args.html_output) if args.html_output else None,
            max_spectrum_plots=args.max_spectrum_plots,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"FFT analyzed {result['run_count']} runs, valid={result['valid_run_count']}")
    print(f"FFT summary: {result['summary_csv']}")
    print(f"Harmonics:  {result['harmonic_csv']}")
    print(f"Sampling:   {result['sampling_csv']}")
    if not args.skip_html:
        print(f"HTML:       {result['html_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
