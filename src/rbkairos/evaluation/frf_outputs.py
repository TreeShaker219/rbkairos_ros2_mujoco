from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


FRF_SELECTED_GAIN_CSV = "frf_selected_gain.csv"
FRF_GAIN_PNG = "frf_gain.png"
FRF_OUTPUT_FILENAMES = (
    FRF_SELECTED_GAIN_CSV,
    FRF_GAIN_PNG,
)
FRF_REQUIRED_POINT_COUNT = 7


def _metric_value(metric: Any, field: str) -> Any:
    if isinstance(metric, Mapping):
        return metric.get(field)
    return getattr(metric, field, None)


def _parameter_value(parameters: Any, field: str) -> float:
    if isinstance(parameters, Mapping):
        value = parameters.get(field)
    else:
        value = getattr(parameters, field, None)
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"FRF output parameter {field} must be finite")
    return numeric


def _format_parameter(value: float) -> str:
    return f"{float(value):g}"


def _finite_sequence(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    try:
        numeric = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in numeric):
        return None
    return numeric


def _collect_rows(
    parameters: Any,
    frf: Any,
    output_dir: Path,
    *,
    frf_reused_from: str | None,
    frf_source_raw_data_paths: list[str] | None,
) -> tuple[dict[str, float], list[dict[str, Any]]] | None:
    if frf is None or _metric_value(frf, "valid") is not True:
        return None
    frequencies = _finite_sequence(_metric_value(frf, "matched_frequencies"))
    gains = _finite_sequence(_metric_value(frf, "gain_db_sim"))
    if frequencies is None or gains is None:
        return None
    if len(frequencies) != FRF_REQUIRED_POINT_COUNT or len(gains) != len(frequencies):
        return None
    if any(frequency <= 0.0 for frequency in frequencies):
        return None
    if len({round(frequency, 12) for frequency in frequencies}) != len(frequencies):
        return None
    if int(_metric_value(frf, "valid_frequency_count") or 0) != FRF_REQUIRED_POINT_COUNT:
        return None
    if int(_metric_value(frf, "rejected_frequency_count") or 0) != 0:
        return None

    parameter_values = {
        field: _parameter_value(parameters, field)
        for field in ("kp", "ki", "i_max")
    }
    summary_path = str(_metric_value(frf, "summary_path") or "")
    source_paths = [str(path) for path in (frf_source_raw_data_paths or [])]
    rows = []
    for frequency, gain_db in sorted(zip(frequencies, gains)):
        rows.append(
            {
                "candidate_id": output_dir.name,
                **parameter_values,
                "f_hz": frequency,
                "fft_gain_db": gain_db,
                "selected_gain_source": "FrfMetric.gain_db_sim",
                "valid": True,
                "frf_reused_from": str(frf_reused_from or ""),
                "summary_path": summary_path,
                "source_raw_data_paths_json": json.dumps(
                    source_paths,
                    sort_keys=True,
                ),
            }
        )
    return parameter_values, rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "candidate_id",
        "kp",
        "ki",
        "i_max",
        "f_hz",
        "fft_gain_db",
        "selected_gain_source",
        "valid",
        "frf_reused_from",
        "summary_path",
        "source_raw_data_paths_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "kp": _format_parameter(row["kp"]),
                    "ki": _format_parameter(row["ki"]),
                    "i_max": _format_parameter(row["i_max"]),
                    "f_hz": repr(row["f_hz"]),
                    "fft_gain_db": repr(row["fft_gain_db"]),
                }
            )


def _plot_payload(
    parameters: dict[str, float],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "parameters": parameters,
        "frf_reused_from": rows[0]["frf_reused_from"],
        "points": [
            {
                "f_hz": row["f_hz"],
                "fft_gain_db": row["fft_gain_db"],
            }
            for row in rows
        ],
    }


def _run_plot_subprocess(payload_path: Path, output_dir: Path) -> None:
    cache_dir = Path(tempfile.gettempdir()) / f"rbkairos_matplotlib_{os.getuid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(cache_dir)
    completed = subprocess.run(
        [
            sys.executable,
            "-s",
            str(Path(__file__).resolve()),
            "--plot-payload",
            str(payload_path),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60.0,
        env=environment,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            "failed to generate FRF gain plot with Matplotlib"
            + (f": {detail}" if detail else "")
        )


def write_frf_candidate_outputs(
    parameters: Any,
    frf: Any,
    output_dir: Path,
    *,
    frf_reused_from: str | None = None,
    frf_source_raw_data_paths: list[str] | None = None,
) -> tuple[Path, ...]:
    """Write the selected seven FRF gains and one candidate-level plot."""
    output_dir = Path(output_dir)
    collected = _collect_rows(
        parameters,
        frf,
        output_dir,
        frf_reused_from=frf_reused_from,
        frf_source_raw_data_paths=frf_source_raw_data_paths,
    )
    if collected is None:
        return ()
    parameter_values, rows = collected
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=output_dir,
        prefix=".frf_outputs_",
    ) as temporary:
        temporary_dir = Path(temporary)
        csv_path = temporary_dir / FRF_SELECTED_GAIN_CSV
        payload_path = temporary_dir / "plot_payload.json"
        _write_csv(csv_path, rows)
        payload_path.write_text(
            json.dumps(
                _plot_payload(parameter_values, rows),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        _run_plot_subprocess(payload_path, temporary_dir)

        for filename in FRF_OUTPUT_FILENAMES:
            generated = temporary_dir / filename
            if not generated.is_file():
                raise RuntimeError(f"missing generated FRF output: {filename}")
            generated.replace(output_dir / filename)

    return tuple(output_dir / filename for filename in FRF_OUTPUT_FILENAMES)


def _unique_gain_ticks(gains: list[float]) -> list[float]:
    ticks: list[float] = []
    for gain in sorted([0.0, *gains]):
        if not any(math.isclose(gain, item, rel_tol=0.0, abs_tol=1e-12) for item in ticks):
            ticks.append(gain)
    return ticks


def _gain_tick_labels(ticks: list[float]) -> list[str]:
    for decimals in range(3, 8):
        labels = ["0" if value == 0.0 else f"{value:.{decimals}f}" for value in ticks]
        if len(labels) == len(set(labels)):
            return labels
    return ["0" if value == 0.0 else f"{value:.10g}" for value in ticks]


def _spread_label_pixels(
    exact_pixels: list[float],
    lower: float,
    upper: float,
    minimum_gap: float,
) -> list[float]:
    if not exact_pixels:
        return []
    adjusted = [max(lower, min(upper, value)) for value in exact_pixels]
    for index in range(1, len(adjusted)):
        adjusted[index] = max(adjusted[index], adjusted[index - 1] + minimum_gap)
    if adjusted[-1] > upper:
        adjusted[-1] = upper
        for index in range(len(adjusted) - 2, -1, -1):
            adjusted[index] = min(adjusted[index], adjusted[index + 1] - minimum_gap)
    if adjusted[0] < lower:
        shift = lower - adjusted[0]
        adjusted = [value + shift for value in adjusted]
    return adjusted


def _write_y_tick_labels(plot: Any, figure: Any, ticks: list[float]) -> None:
    from matplotlib.transforms import blended_transform_factory

    figure.canvas.draw()
    labels = _gain_tick_labels(ticks)
    exact_pixels = [plot.transData.transform((0.0, value))[1] for value in ticks]
    adjusted_pixels = _spread_label_pixels(
        exact_pixels,
        plot.bbox.y0 + 8.0,
        plot.bbox.y1 - 8.0,
        22.0,
    )
    inverse = plot.transData.inverted()
    transform = blended_transform_factory(plot.transAxes, plot.transData)
    for tick, label, exact_pixel, adjusted_pixel in zip(
        ticks,
        labels,
        exact_pixels,
        adjusted_pixels,
    ):
        adjusted_data_y = inverse.transform((0.0, adjusted_pixel))[1]
        if abs(adjusted_pixel - exact_pixel) > 1.0:
            plot.plot(
                [-0.012, 0.0],
                [adjusted_data_y, tick],
                transform=transform,
                color="#6B7280",
                linewidth=0.8,
                clip_on=False,
            )
        plot.text(
            -0.022,
            adjusted_data_y,
            label,
            transform=transform,
            ha="right",
            va="center",
            fontsize=13,
            color="#374151",
            clip_on=False,
        )


def _write_gain_plot(plt: Any, target: Path, payload: dict[str, Any]) -> None:
    parameters = payload["parameters"]
    points = payload["points"]
    frequencies = [float(point["f_hz"]) for point in points]
    gains = [float(point["fft_gain_db"]) for point in points]
    reused_from = str(payload.get("frf_reused_from") or "")

    figure, plot = plt.subplots(figsize=(12, 8), dpi=100)
    figure.patch.set_facecolor("white")
    plot.set_facecolor("white")
    plot.plot(
        frequencies,
        gains,
        color="#2563EB",
        linewidth=3.0,
        marker="o",
        markersize=9,
        markerfacecolor="#2563EB",
        markeredgecolor="white",
        markeredgewidth=1.5,
    )
    plot.set_title(
        "X-direction FRF FFT gain",
        fontsize=24,
        fontweight="bold",
        pad=48 if reused_from else 34,
    )
    plot.text(
        0.5,
        1.045 if reused_from else 1.035,
        (
            f"Kp={_format_parameter(parameters['kp'])}, "
            f"Ki={_format_parameter(parameters['ki'])}, "
            f"i_max={_format_parameter(parameters['i_max'])} N m"
        ),
        transform=plot.transAxes,
        ha="center",
        va="bottom",
        fontsize=16,
        color="#374151",
    )
    if reused_from:
        plot.text(
            0.5,
            1.005,
            f"FRF reused from {reused_from}",
            transform=plot.transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
            color="#6B7280",
        )
    plot.set_xlabel(
        "Frequency (Hz, log scale)",
        fontsize=18,
        fontweight="bold",
        labelpad=18,
    )
    plot.set_ylabel("FFT gain (dB)", fontsize=18, fontweight="bold", labelpad=50)
    plot.set_xscale("log")
    plot.set_xticks(frequencies, [f"{frequency:.2f}" for frequency in frequencies])
    plot.set_xlim(min(frequencies) / 1.15, max(frequencies) * 1.15)

    gain_ticks = _unique_gain_ticks(gains)
    plot.set_yticks(gain_ticks)
    plot.set_yticklabels([])
    y_min = min(gain_ticks)
    y_max = max(gain_ticks)
    y_span = max(1.0, y_max - y_min)
    plot.set_ylim(y_min - 0.08 * y_span, y_max + 0.14 * y_span)
    plot.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.75)
    plot.grid(axis="x", visible=False)
    plot.tick_params(
        axis="both",
        labelsize=14,
        colors="#374151",
        length=6,
        width=1.2,
    )
    for spine in plot.spines.values():
        spine.set_color("#6B7280")
        spine.set_linewidth(1.2)

    figure.subplots_adjust(
        left=0.18,
        right=0.96,
        bottom=0.16,
        top=0.76 if reused_from else 0.80,
    )
    _write_y_tick_labels(plot, figure, gain_ticks)
    figure.savefig(target, dpi=100, facecolor="white")
    plt.close(figure)


def _write_plot_from_payload(payload_path: Path, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    _write_gain_plot(plt, output_dir / FRF_GAIN_PNG, payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-payload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _write_plot_from_payload(args.plot_payload, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
