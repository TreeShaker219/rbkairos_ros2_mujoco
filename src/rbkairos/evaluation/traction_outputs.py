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


TRACTION_SELECTED_FORCE_CSV = "traction_selected_force.csv"
TRACTION_FORCE_X_PNG = "traction_force_x.png"
TRACTION_FORCE_Y_PNG = "traction_force_y.png"
TRACTION_OUTPUT_FILENAMES = (
    TRACTION_SELECTED_FORCE_CSV,
    TRACTION_FORCE_X_PNG,
    TRACTION_FORCE_Y_PNG,
)


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
        raise ValueError(f"traction output parameter {field} must be finite")
    return numeric


def _format_parameter(value: float) -> str:
    return f"{float(value):g}"


def _collect_rows(
    parameters: Any,
    traction: Any,
    output_dir: Path,
) -> tuple[dict[str, float], list[dict[str, Any]]] | None:
    from evaluation.traction_conditions import TRACTION_CONDITIONS

    if not isinstance(traction, Mapping):
        return None
    canonical_ids = tuple(item.condition_id for item in TRACTION_CONDITIONS)
    if set(traction) != set(canonical_ids):
        return None

    parameter_values = {
        field: _parameter_value(parameters, field)
        for field in ("kp", "ki", "i_max")
    }
    rows: list[dict[str, Any]] = []
    for condition in TRACTION_CONDITIONS:
        metric = traction[condition.condition_id]
        if _metric_value(metric, "valid") is not True:
            return None
        try:
            force = float(_metric_value(metric, "F_traction_sim"))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(force):
            return None
        rows.append(
            {
                "candidate_id": output_dir.name,
                **parameter_values,
                "condition_id": condition.condition_id,
                "axis": condition.axis,
                "velocity_component": f"v{condition.axis}",
                "velocity_mps": float(condition.speed_mps),
                "selected_force_n": force,
                "selected_force_source": "steady_tow_force_n",
                "valid": True,
                "summary_path": str(_metric_value(metric, "summary_path") or ""),
            }
        )
    return parameter_values, rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "candidate_id",
        "kp",
        "ki",
        "i_max",
        "condition_id",
        "axis",
        "velocity_component",
        "velocity_mps",
        "selected_force_n",
        "selected_force_source",
        "valid",
        "summary_path",
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
                    "selected_force_n": repr(row["selected_force_n"]),
                }
            )


def _plot_payload(
    parameters: dict[str, float],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "parameters": parameters,
        "axes": {
            axis: [
                {
                    "velocity_mps": row["velocity_mps"],
                    "selected_force_n": row["selected_force_n"],
                }
                for row in rows
                if row["axis"] == axis
            ]
            for axis in ("x", "y")
        },
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
            "failed to generate Tendon force plots with Matplotlib"
            + (f": {detail}" if detail else "")
        )


def write_traction_candidate_outputs(
    parameters: Any,
    traction: Any,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Write the six selected forces and two axis plots for a complete candidate."""
    output_dir = Path(output_dir)
    collected = _collect_rows(parameters, traction, output_dir)
    if collected is None:
        return ()
    parameter_values, rows = collected
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=output_dir,
        prefix=".traction_outputs_",
    ) as temporary:
        temporary_dir = Path(temporary)
        csv_path = temporary_dir / TRACTION_SELECTED_FORCE_CSV
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

        for filename in TRACTION_OUTPUT_FILENAMES:
            generated = temporary_dir / filename
            if not generated.is_file():
                raise RuntimeError(f"missing generated Tendon output: {filename}")
            generated.replace(output_dir / filename)

    return tuple(output_dir / filename for filename in TRACTION_OUTPUT_FILENAMES)


def _unique_force_ticks(forces: list[float]) -> list[float]:
    ticks = [0.0]
    for force in sorted(forces):
        if not any(math.isclose(force, item, rel_tol=0.0, abs_tol=1e-12) for item in ticks):
            ticks.append(force)
    return ticks


def _force_tick_labels(ticks: list[float]) -> list[str]:
    nonzero = ticks[1:]
    for decimals in range(3, 7):
        labels = [f"{value:.{decimals}f}" for value in nonzero]
        if len(labels) == len(set(labels)):
            return ["0", *labels]
    return ["0", *[f"{value:.9g}" for value in nonzero]]


def _write_axis_plot(plt: Any, target: Path, payload: dict[str, Any], axis: str) -> None:
    parameters = payload["parameters"]
    points = payload["axes"][axis]
    speeds = [float(point["velocity_mps"]) for point in points]
    forces = [float(point["selected_force_n"]) for point in points]
    color = "#2563EB" if axis == "x" else "#D97706"

    figure, plot = plt.subplots(figsize=(12, 8), dpi=100)
    figure.patch.set_facecolor("white")
    plot.set_facecolor("white")
    plot.plot(
        speeds,
        forces,
        color=color,
        linewidth=3.0,
        marker="o",
        markersize=9,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=1.5,
    )
    plot.set_title(
        f"{axis.upper()}-direction Tendon force",
        fontsize=24,
        fontweight="bold",
        pad=34,
    )
    plot.text(
        0.5,
        1.035,
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
    plot.set_xlabel(
        rf"$v_{axis}$ (m/s)",
        fontsize=18,
        fontweight="bold",
        labelpad=18,
    )
    plot.set_ylabel("F (N)", fontsize=18, fontweight="bold", labelpad=18)
    plot.set_xticks(speeds, [f"{speed:.2f}" for speed in speeds])
    plot.set_xlim(0.0, max(0.11, max(speeds) * 1.1))

    force_ticks = _unique_force_ticks(forces)
    plot.set_yticks(force_ticks, _force_tick_labels(force_ticks))
    max_force = max(forces)
    plot.set_ylim(0.0, max(1.0, max_force * 1.16))
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

    figure.subplots_adjust(left=0.14, right=0.96, bottom=0.16, top=0.80)
    figure.savefig(target, dpi=100, facecolor="white")
    plt.close(figure)


def _write_plots_from_payload(payload_path: Path, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    for axis, filename in (
        ("x", TRACTION_FORCE_X_PNG),
        ("y", TRACTION_FORCE_Y_PNG),
    ):
        _write_axis_plot(plt, output_dir / filename, payload, axis)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-payload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _write_plots_from_payload(args.plot_payload, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
