#!/usr/bin/env python3
"""CLI for single-run and batch traction-tendon analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from traction_tendon_analysis import (
    AnalysisInputError,
    analyze_traction_tendon_batch,
    analyze_traction_tendon_run,
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Tendon run directory or batch root")
    parser.add_argument(
        "--batch", action="store_true", help="discover and analyze all Tendon runs"
    )
    parser.add_argument("--axis", choices=("x", "y"), help="require this axis for a single run")
    parser.add_argument("--steady-window", type=float, default=2.0)
    parser.add_argument("--steady-window-step", type=float, default=0.25)
    parser.add_argument("--min-observation", type=float, default=2.0)
    parser.add_argument("--required-stable-windows", type=int, default=4)
    parser.add_argument("--max-wait", type=float, default=60.0)
    parser.add_argument("--min-active-ratio", type=float, default=0.95)
    parser.add_argument("--max-force-cv", type=float, default=0.05)
    parser.add_argument("--max-force-slope", type=float, default=0.5)
    parser.add_argument("--max-constraint-extension", type=float, default=0.005)
    parser.add_argument("--max-steady-base-velocity", type=float, default=0.005)
    parser.add_argument("--max-i-term-slope", type=float, default=0.005)
    parser.add_argument("--min-i-clamp-ratio", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, help="batch output directory")
    args = parser.parse_args(argv)
    for name in (
        "steady_window", "steady_window_step", "min_observation", "max_wait"
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.required_stable_windows <= 0:
        parser.error("--required-stable-windows must be positive")
    if not 0.0 <= args.min_active_ratio <= 1.0:
        parser.error("--min-active-ratio must be in [0, 1]")
    if not 0.0 <= args.min_i_clamp_ratio <= 1.0:
        parser.error("--min-i-clamp-ratio must be in [0, 1]")
    for name in (
        "max_force_cv",
        "max_force_slope",
        "max_constraint_extension",
        "max_steady_base_velocity",
        "max_i_term_slope",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    options = {
        "steady_window_s": args.steady_window,
        "steady_window_step_s": args.steady_window_step,
        "min_observation_s": args.min_observation,
        "required_stable_windows": args.required_stable_windows,
        "max_wait_s": args.max_wait,
        "min_tendon_active_ratio": args.min_active_ratio,
        "max_force_cv": args.max_force_cv,
        "max_force_slope_n_s": args.max_force_slope,
        "max_constraint_extension_m": args.max_constraint_extension,
        "max_steady_base_velocity_mps": args.max_steady_base_velocity,
        "max_i_term_slope_nm_s": args.max_i_term_slope,
        "min_i_clamp_ratio": args.min_i_clamp_ratio,
    }
    try:
        if args.batch:
            summaries = analyze_traction_tendon_batch(
                args.input_dir,
                output_dir=args.output_dir,
                **options,
            )
            print(json.dumps({"run_count": len(summaries)}, sort_keys=True))
        else:
            summary = analyze_traction_tendon_run(
                args.input_dir,
                requested_axis=args.axis,
                **options,
            )
            selected = {
                key: summary[key]
                for key in ("axis", "steady_tow_force_n", "valid", "reject_reason")
            }
            print(json.dumps(selected, sort_keys=True))
            if not summary["valid"]:
                return 1
    except AnalysisInputError as exc:
        print(f"traction tendon analysis failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
