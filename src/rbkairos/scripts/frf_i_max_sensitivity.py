#!/usr/bin/env python3
"""Pure-Python plan, diagnostics, and reports for the FRF i_max sensitivity pilot."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence


PILOT_SCHEMA_VERSION = 1
WHEEL_DOF_NAMES = (
    "left_front_wheel_joint",
    "right_front_wheel_joint",
    "left_back_wheel_joint",
    "right_back_wheel_joint",
)
PID_CSV_FIELDS = (
    "t_experiment",
    "header_stamp_s",
    "joint_name",
    "velocity_target",
    "velocity_actual",
    "velocity_error",
    "p_term",
    "i_term",
    "d_term",
    "pid_raw_output",
    "effort_clamped_output",
    "qfrc_applied",
    "i_clamp_active",
    "effort_clamp_active",
    "kp",
    "ki",
    "kd",
    "i_max",
    "i_min",
    "antiwindup",
    "effort_limit_nm",
)
SUMMARY_FIELDS = (
    "candidate_id",
    "kp",
    "ki",
    "i_max",
    "frequency_hz",
    "gain_db_sim",
    "gain_db_real",
    "gain_error_db",
    "squared_gain_error_db",
    "phase_deg_sim",
    "thd_2_to_max",
    "valid",
    "reject_reason",
    "pid_sample_rows",
    "max_abs_i_term_nm",
    "i_bound_usage_ratio",
    "all_wheel_i_clamp_ratio",
    "any_wheel_effort_clamp_ratio",
    "i_clamp_any",
    "effort_clamp_any",
)


class PilotInputError(ValueError):
    """Raised when a pilot artifact violates the frozen contract."""


def _token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _candidate_id(i_max: float) -> str:
    return f"kp50_ki1_imax{_token(i_max)}"


def default_pilot_plan() -> dict[str, Any]:
    """Freeze a full seven-frequency, no-repeat, three-bound comparison."""
    candidates = [
        {
            "candidate_id": _candidate_id(i_max),
            "kp": 50.0,
            "ki": 1.0,
            "kd": 0.0,
            "i_max": i_max,
            "i_min": -i_max,
            "effort_limit_nm": 100.0,
        }
        for i_max in (0.5, 1.5, 3.0)
    ]
    frequencies = [0.2, 0.3, 0.5, 1.0, 1.8, 2.8, 4.0]
    runs = []
    for candidate in candidates:
        for frequency in frequencies:
            runs.append(
                {
                    "run_id": f"{candidate['candidate_id']}_f{_token(frequency)}hz_r01",
                    "candidate_id": candidate["candidate_id"],
                    "frequency_hz": frequency,
                    "replicate": 1,
                }
            )
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "experiment": "frf_i_max_sensitivity_pilot",
        "status": "planned_before_runtime",
        "fixed": {
            "mu_low": 0.01,
            "mu_high": 0.7,
            "effort_limit_nm": 100.0,
            "kd": 0.0,
            "ur5_preset": "up",
            "amplitude_mps": 0.10,
        },
        "protocol": {
            "cycles": 6.0,
            "ramp_cycles": 1.0,
            "discard_cycles": 2.0,
            "analysis_cycles": 4,
            "fresh_simulator_per_frequency": True,
            "replicates": 1,
            "primary_cost": "mean squared gain error in dB, identical to J_frf",
            "classification": (
                "No retrospective pass threshold. Report gain/J_frf spans together with "
                "observed I-clamp and effort-clamp state."
            ),
        },
        "candidates": candidates,
        "frequencies_hz": frequencies,
        "runs": runs,
    }


def write_pilot_plan(plan: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pilot_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "pilot_matrix.csv").open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=("run_id", "candidate_id", "frequency_hz", "replicate"),
        )
        writer.writeheader()
        writer.writerows(plan["runs"])


def _finite(raw: Any, label: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise PilotInputError(f"{label} must be numeric") from exc
    if not math.isfinite(value):
        raise PilotInputError(f"{label} must be finite")
    return value


def _boolean(raw: Any, label: str) -> bool:
    text = str(raw).strip().lower()
    if text in {"1", "true"}:
        return True
    if text in {"0", "false"}:
        return False
    raise PilotInputError(f"{label} must be 0/1 or true/false")


def _read_csv(path: Path, required: Sequence[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise PilotInputError(f"missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise PilotInputError(f"{path} missing columns: {sorted(missing)}")
        return list(reader)


def summarize_pid_window(
    path: Path,
    *,
    analysis_start_s: float,
    analysis_end_s: float,
    expected: Mapping[str, float],
) -> dict[str, Any]:
    """Summarize exact D1 PID diagnostics over the FFT analysis interval."""
    start = _finite(analysis_start_s, "analysis_start_s")
    end = _finite(analysis_end_s, "analysis_end_s")
    if end <= start:
        raise PilotInputError("analysis_end_s must exceed analysis_start_s")
    rows = _read_csv(path, PID_CSV_FIELDS)
    selected: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=2):
        t = _finite(row["t_experiment"], f"{path}:{row_index}:t_experiment")
        if t < start - 1e-9 or t > end + 1e-9:
            continue
        joint = str(row["joint_name"])
        if joint not in WHEEL_DOF_NAMES:
            raise PilotInputError(f"{path}:{row_index}:unexpected joint {joint!r}")
        parsed = {"t": t, "joint": joint}
        for field in (
            "i_term",
            "qfrc_applied",
            "kp",
            "ki",
            "kd",
            "i_max",
            "i_min",
            "effort_limit_nm",
        ):
            parsed[field] = _finite(row[field], f"{path}:{row_index}:{field}")
        parsed["i_clamp_active"] = _boolean(
            row["i_clamp_active"], f"{path}:{row_index}:i_clamp_active"
        )
        parsed["effort_clamp_active"] = _boolean(
            row["effort_clamp_active"], f"{path}:{row_index}:effort_clamp_active"
        )
        for field in ("kp", "ki", "kd", "i_max"):
            target = _finite(expected[field], f"expected.{field}")
            if not math.isclose(parsed[field], target, rel_tol=0.0, abs_tol=1e-9):
                raise PilotInputError(
                    f"{path}:{row_index}:{field}={parsed[field]} != expected {target}"
                )
        if not math.isclose(
            parsed["i_min"], -_finite(expected["i_max"], "expected.i_max"),
            rel_tol=0.0, abs_tol=1e-9,
        ):
            raise PilotInputError(f"{path}:{row_index}:i_min contract mismatch")
        selected.append(parsed)
    if not selected:
        raise PilotInputError(f"{path} has no PID samples in FFT window [{start}, {end}]")

    per_wheel: dict[str, dict[str, Any]] = {}
    for wheel in WHEEL_DOF_NAMES:
        wheel_rows = [row for row in selected if row["joint"] == wheel]
        if not wheel_rows:
            raise PilotInputError(f"{path} has no FFT-window PID samples for {wheel}")
        per_wheel[wheel] = {
            "sample_rows": len(wheel_rows),
            "max_abs_i_term_nm": max(abs(row["i_term"]) for row in wheel_rows),
            "i_clamp_ratio": sum(row["i_clamp_active"] for row in wheel_rows) / len(wheel_rows),
            "effort_clamp_ratio": (
                sum(row["effort_clamp_active"] for row in wheel_rows) / len(wheel_rows)
            ),
            "max_abs_qfrc_applied_nm": max(
                abs(row["qfrc_applied"]) for row in wheel_rows
            ),
        }

    by_time: dict[float, list[dict[str, Any]]] = {}
    for row in selected:
        by_time.setdefault(row["t"], []).append(row)
    effort_any_ratio = sum(
        any(row["effort_clamp_active"] for row in at_time)
        for at_time in by_time.values()
    ) / len(by_time)
    return {
        "analysis_start_s": start,
        "analysis_end_s": end,
        "sample_rows": len(selected),
        "max_abs_i_term_nm": max(abs(row["i_term"]) for row in selected),
        "all_wheel_i_clamp_ratio": min(
            values["i_clamp_ratio"] for values in per_wheel.values()
        ),
        "any_wheel_effort_clamp_ratio": effort_any_ratio,
        "i_clamp_any": any(row["i_clamp_active"] for row in selected),
        "effort_clamp_any": any(row["effort_clamp_active"] for row in selected),
        "per_wheel": per_wheel,
    }


def real_gain_map(path: Path) -> dict[float, float]:
    rows = _read_csv(path, ("f_hz", "valid"))
    result = {}
    for index, row in enumerate(rows, start=2):
        if not _boolean(row["valid"], f"{path}:{index}:valid"):
            continue
        field = "fft_gain_db" if row.get("fft_gain_db", "") != "" else "gain_db"
        if field not in row:
            raise PilotInputError(f"{path} has no FFT or legacy gain dB column")
        result[_finite(row["f_hz"], f"{path}:{index}:f_hz")] = _finite(
            row[field], f"{path}:{index}:{field}"
        )
    if not result:
        raise PilotInputError(f"{path} has no valid real FRF rows")
    return result


def build_candidate_rows(
    candidate: Mapping[str, Any],
    fft_summary_path: Path,
    pid_paths: Mapping[float, Path],
    real_gains: Mapping[float, float],
) -> list[dict[str, Any]]:
    fft_rows = _read_csv(
        fft_summary_path,
        ("f_hz", "analysis_start_s", "analysis_end_s", "fft_gain_db", "valid"),
    )
    result = []
    for row_index, fft_row in enumerate(fft_rows, start=2):
        frequency = _finite(fft_row["f_hz"], f"{fft_summary_path}:{row_index}:f_hz")
        if frequency not in pid_paths or frequency not in real_gains:
            raise PilotInputError(f"missing PID or real reference for {frequency:g} Hz")
        pid = summarize_pid_window(
            pid_paths[frequency],
            analysis_start_s=_finite(fft_row["analysis_start_s"], "analysis_start_s"),
            analysis_end_s=_finite(fft_row["analysis_end_s"], "analysis_end_s"),
            expected=candidate,
        )
        valid = _boolean(fft_row["valid"], f"{fft_summary_path}:{row_index}:valid")
        sim_gain = _finite(fft_row["fft_gain_db"], "fft_gain_db")
        real_gain = real_gains[frequency]
        error = sim_gain - real_gain
        result.append(
            {
                "candidate_id": candidate["candidate_id"],
                "kp": float(candidate["kp"]),
                "ki": float(candidate["ki"]),
                "i_max": float(candidate["i_max"]),
                "frequency_hz": frequency,
                "gain_db_sim": sim_gain,
                "gain_db_real": real_gain,
                "gain_error_db": error,
                "squared_gain_error_db": error * error,
                "phase_deg_sim": _finite(fft_row.get("fft_phase_deg"), "fft_phase_deg"),
                "thd_2_to_max": _finite(fft_row.get("thd_2_to_max"), "thd_2_to_max"),
                "valid": valid and not pid["effort_clamp_any"],
                "reject_reason": (
                    str(fft_row.get("reject_reason", ""))
                    if valid
                    else str(fft_row.get("reject_reason", "invalid_fft"))
                ),
                "pid_sample_rows": pid["sample_rows"],
                "max_abs_i_term_nm": pid["max_abs_i_term_nm"],
                "i_bound_usage_ratio": pid["max_abs_i_term_nm"] / float(candidate["i_max"]),
                "all_wheel_i_clamp_ratio": pid["all_wheel_i_clamp_ratio"],
                "any_wheel_effort_clamp_ratio": pid["any_wheel_effort_clamp_ratio"],
                "i_clamp_any": pid["i_clamp_any"],
                "effort_clamp_any": pid["effort_clamp_any"],
                "pid_per_wheel": pid["per_wheel"],
            }
        )
    return result


def _svg_polyline(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[tuple[str, str, Sequence[tuple[float, float]]]],
) -> None:
    width, height = 900, 520
    left, right, top, bottom = 80, 30, 55, 70
    points = [point for _, _, values in series for point in values]
    if not points:
        raise PilotInputError(f"cannot plot empty series for {title}")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if math.isclose(xmin, xmax):
        xmin, xmax = xmin - 0.5, xmax + 0.5
    if math.isclose(ymin, ymax):
        ymin, ymax = ymin - 0.5, ymax + 0.5
    ypad = max((ymax - ymin) * 0.08, 1e-6)
    ymin, ymax = ymin - ypad, ymax + ypad
    plot_w, plot_h = width - left - right, height - top - bottom

    def px(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * plot_w

    def py(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * plot_h

    body = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111"/>',
        (
            f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" '
            f'y2="{top+plot_h}" stroke="#111"/>'
        ),
        f'<text x="{left+plot_w/2}" y="{height-18}" text-anchor="middle">{x_label}</text>',
        (
            f'<text x="18" y="{top+plot_h/2}" text-anchor="middle" '
            f'transform="rotate(-90 18 {top+plot_h/2})">{y_label}</text>'
        ),
        f'<text x="{left-8}" y="{py(ymax)+5}" text-anchor="end" font-size="11">{ymax:.4g}</text>',
        f'<text x="{left-8}" y="{py(ymin)+5}" text-anchor="end" font-size="11">{ymin:.4g}</text>',
    ]
    for index, (label, color, values) in enumerate(series):
        coords = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in values)
        body.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in values:
            body.append(f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="3" fill="{color}"/>')
        legend_y = top + 18 * index
        body.append(
            f'<line x1="{width-210}" y1="{legend_y}" x2="{width-190}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        body.append(f'<text x="{width-183}" y="{legend_y+4}" font-size="12">{label}</text>')
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        + "".join(body)
        + "</svg>\n",
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_pilot_reports(rows: Iterable[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    """Write machine-readable summaries and compact SVG sensitivity plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = [dict(row) for row in rows]
    if not normalized:
        raise PilotInputError("pilot report requires at least one row")
    for row in normalized:
        default_error = float(row["gain_db_sim"]) - float(row["gain_db_real"])
        gain_error = float(row.get("gain_error_db", default_error))
        row.setdefault("gain_error_db", gain_error)
        row.setdefault("squared_gain_error_db", gain_error * gain_error)
        row.setdefault("reject_reason", "")
        row.setdefault("pid_sample_rows", 0)
        row.setdefault("effort_clamp_any", False)
        row.setdefault("thd_2_to_max", 0.0)
        row.setdefault(
            "i_bound_usage_ratio",
            float(row["max_abs_i_term_nm"]) / float(row["i_max"]),
        )
    normalized.sort(key=lambda row: (float(row["i_max"]), float(row["frequency_hz"])))

    with (output_dir / "frf_i_max_sensitivity_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized)

    candidates = []
    i_max_values = sorted({float(row["i_max"]) for row in normalized})
    for i_max in i_max_values:
        candidate_rows = [row for row in normalized if float(row["i_max"]) == i_max]
        valid_rows = [row for row in candidate_rows if bool(row["valid"])]
        candidates.append(
            {
                "candidate_id": candidate_rows[0]["candidate_id"],
                "kp": float(candidate_rows[0]["kp"]),
                "ki": float(candidate_rows[0]["ki"]),
                "i_max": i_max,
                "frequency_count": len(candidate_rows),
                "valid_frequency_count": len(valid_rows),
                "J_frf": (
                    statistics.fmean(
                        [float(row["squared_gain_error_db"]) for row in valid_rows]
                    )
                    if valid_rows else None
                ),
                "max_abs_i_term_nm": max(
                    float(row["max_abs_i_term_nm"]) for row in candidate_rows
                ),
                "max_i_bound_usage_ratio": max(
                    float(row["i_bound_usage_ratio"]) for row in candidate_rows
                ),
                "i_clamp_frequency_count": sum(bool(row["i_clamp_any"]) for row in candidate_rows),
                "effort_clamp_frequency_count": sum(
                    bool(row["effort_clamp_any"]) for row in candidate_rows
                ),
            }
        )

    per_frequency = []
    frequencies = sorted({float(row["frequency_hz"]) for row in normalized})
    for frequency in frequencies:
        frequency_rows = [
            row for row in normalized if float(row["frequency_hz"]) == frequency
        ]
        gains = [float(row["gain_db_sim"]) for row in frequency_rows]
        phases = [float(row["phase_deg_sim"]) for row in frequency_rows]
        per_frequency.append(
            {
                "frequency_hz": frequency,
                "gain_span_db": max(gains) - min(gains),
                "phase_span_deg": max(phases) - min(phases),
                "i_clamp_i_max_values": [
                    float(row["i_max"]) for row in frequency_rows if bool(row["i_clamp_any"])
                ],
                "effort_clamp_i_max_values": [
                    float(row["i_max"])
                    for row in frequency_rows
                    if bool(row["effort_clamp_any"])
                ],
            }
        )

    report = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "experiment": "frf_i_max_sensitivity_pilot",
        "row_count": len(normalized),
        "candidate_count": len(candidates),
        "frequency_count": len(frequencies),
        "candidates": candidates,
        "per_frequency": per_frequency,
        "global": {
            "J_frf_span": (
                max(float(item["J_frf"]) for item in candidates if item["J_frf"] is not None)
                - min(float(item["J_frf"]) for item in candidates if item["J_frf"] is not None)
            ),
            "max_gain_span_db": max(item["gain_span_db"] for item in per_frequency),
            "max_phase_span_deg": max(item["phase_span_deg"] for item in per_frequency),
            "any_i_clamp": any(bool(row["i_clamp_any"]) for row in normalized),
            "any_effort_clamp": any(bool(row["effort_clamp_any"]) for row in normalized),
        },
        "method_note": (
            "J_frf is the mean squared real-vs-sim gain error in dB. I-clamp is "
            "descriptive causal evidence; no retrospective sensitivity threshold is applied."
        ),
    }
    (output_dir / "frf_i_max_sensitivity_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    colors = ("#2563eb", "#d97706", "#059669", "#6b7280")
    gain_series = []
    for index, i_max in enumerate(i_max_values):
        values = [
            (float(row["frequency_hz"]), float(row["gain_db_sim"]))
            for row in normalized
            if float(row["i_max"]) == i_max
        ]
        gain_series.append((f"i_max={i_max:g}", colors[index], values))
    real_values = []
    for frequency in frequencies:
        first = next(row for row in normalized if float(row["frequency_hz"]) == frequency)
        real_values.append((frequency, float(first["gain_db_real"])))
    gain_series.append(("real", colors[-1], real_values))
    _svg_polyline(
        output_dir / "frf_i_max_gain_overlay.svg",
        "FRF gain by i_max",
        "frequency (Hz)",
        "gain (dB)",
        gain_series,
    )
    _svg_polyline(
        output_dir / "frf_i_max_gain_span.svg",
        "Gain span across i_max candidates",
        "frequency (Hz)",
        "max-min gain (dB)",
        [
            (
                "gain span",
                "#7c3aed",
                [(item["frequency_hz"], item["gain_span_db"]) for item in per_frequency],
            )
        ],
    )
    _svg_polyline(
        output_dir / "frf_i_max_integral_activity.svg",
        "Integral activity in FFT window",
        "frequency (Hz)",
        "I-clamp ratio",
        [
            (
                f"clamp ratio i_max={i_max:g}",
                colors[index],
                [
                    (float(row["frequency_hz"]), float(row["all_wheel_i_clamp_ratio"]))
                    for row in normalized
                    if float(row["i_max"]) == i_max
                ],
            )
            for index, i_max in enumerate(i_max_values)
        ],
    )
    _svg_polyline(
        output_dir / "frf_i_max_integral_usage.svg",
        "Integral bound usage in FFT window",
        "frequency (Hz)",
        "max abs(I-term) / i_max",
        [
            (
                f"i_max={i_max:g}",
                colors[index],
                [
                    (float(row["frequency_hz"]), float(row["i_bound_usage_ratio"]))
                    for row in normalized
                    if float(row["i_max"]) == i_max
                ],
            )
            for index, i_max in enumerate(i_max_values)
        ],
    )
    _svg_polyline(
        output_dir / "frf_i_max_cost.svg",
        "J_frf by i_max",
        "i_max (Nm)",
        "J_frf (dB^2)",
        [("J_frf", "#dc2626", [(item["i_max"], item["J_frf"]) for item in candidates])],
    )
    return report
