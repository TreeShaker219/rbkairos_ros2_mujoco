#!/usr/bin/env python3
"""Single entry point for RB-KAIROS parameter evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from evaluation.config import EvaluationConfig, load_config  # noqa: E402
from evaluation.costs import frozen_cost_contract  # noqa: E402
from evaluation.campaign import (  # noqa: E402
    build_candidate_manifest,
    build_initial_campaign_state,
    load_campaign_spec,
)
from evaluation.campaign_executor import (  # noqa: E402
    CampaignExecutionError,
    execute_managed_campaign,
)
from evaluation.managed import build_managed_dry_run, evaluate_managed  # noqa: E402
from evaluation.parameters import (  # noqa: E402
    EvaluationParameters,
    fixed_parameters_dict,
)
from evaluation.traction_conditions import TRACTION_CONDITION_IDS  # noqa: E402
from evaluation.result import evaluate_offline  # noqa: E402


def nested_override(path: list[str], value: Any) -> dict[str, Any]:
    root: dict[str, Any] = {}
    cursor = root
    for key in path[:-1]:
        cursor[key] = {}
        cursor = cursor[key]
    cursor[path[-1]] = value
    return root


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def parse_condition_assignment(raw: str, *, value_kind: str) -> tuple[str, Any]:
    condition_id, separator, value = str(raw).partition("=")
    if not separator or condition_id not in TRACTION_CONDITION_IDS or not value:
        raise argparse.ArgumentTypeError(
            "expected CONDITION=VALUE with CONDITION in "
            + ",".join(TRACTION_CONDITION_IDS)
        )
    if value_kind == "float":
        try:
            return condition_id, float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid numeric value: {value}") from exc
    if value_kind == "path":
        return condition_id, str(Path(value))
    raise ValueError(f"unsupported assignment kind: {value_kind}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RB-KAIROS parameters theta=(Kp, Ki, i_max)."
    )
    parser.add_argument(
        "--campaign-config",
        type=Path,
        default=None,
        help=(
            "candidates/grid YAML; defaults to dry-run and "
            "cannot be combined with --kp/--ki/--i-max"
        ),
    )
    parser.add_argument("--kp", type=float, default=None, help="wheel velocity PID Kp")
    parser.add_argument("--ki", type=float, default=None, help="wheel velocity PID Ki")
    parser.add_argument(
        "--i-max",
        type=float,
        default=None,
        help="symmetric wheel velocity PID integral bound; i_min=-i_max",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config path; defaults to src/rbkairos/config/evaluation/default.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=None, help="override output root")
    parser.add_argument(
        "--reuse-policy",
        choices=["fresh", "explicit"],
        default=None,
        help="result reuse policy; v1 computes fresh unless future explicit reuse is added",
    )
    parser.add_argument(
        "--traction-reference",
        action="append",
        default=[],
        metavar="CONDITION=FORCE_N",
    )
    parser.add_argument(
        "--traction-sim-run",
        action="append",
        default=[],
        metavar="CONDITION=RUN_DIR",
    )
    parser.add_argument("--frf-sim-summary", type=Path, default=None)
    parser.add_argument(
        "--frf-real-reference",
        type=Path,
        default=None,
        help="frozen real FRF mean/std/weight reference CSV",
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "managed"],
        default=None,
        help=(
            "evaluation mode: offline reads existing data; managed launches MuJoCo "
            "and recorders"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "validate parameters/config and print the resolved evaluation request "
            "without computing metrics"
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="explicitly authorize managed campaign execution",
    )
    parser.add_argument(
        "--experiment-scope",
        choices=["all", "tendon", "frf"],
        default="all",
        help=(
            "campaign experiment family to execute; tendon and frf scopes may be "
            "run separately against the same campaign state"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="enable strict analyzer checks where supported",
    )
    parser.add_argument(
        "--no-runtime-assets",
        action="store_true",
        help=(
            "skip candidate-local URDF/MJCF/Traction scene/config asset generation "
            "in offline mode"
        ),
    )
    args = parser.parse_args(argv)
    single_values = (args.kp, args.ki, args.i_max)
    if args.campaign_config is not None:
        if any(value is not None for value in single_values):
            parser.error(
                "--campaign-config cannot be combined with --kp/--ki/--i-max"
            )
        if args.dry_run and args.execute:
            parser.error("--dry-run and --execute are mutually exclusive")
        if not args.execute:
            args.dry_run = True
        if args.mode not in (None, "managed"):
            parser.error("--campaign-config supports only managed mode")
        if args.traction_reference or args.traction_sim_run:
            parser.error("campaign mode cannot use per-run Traction overrides")
        if args.frf_sim_summary is not None or args.frf_real_reference is not None:
            parser.error("campaign mode cannot use offline FRF overrides")
        if args.no_runtime_assets:
            parser.error("campaign mode cannot disable runtime assets")
        if args.strict and not args.execute:
            parser.error("campaign dry-run cannot use --strict")
    else:
        if args.execute:
            parser.error("--execute requires --campaign-config")
        if args.experiment_scope != "all":
            parser.error("--experiment-scope requires --campaign-config")
        if any(value is None for value in single_values):
            parser.error("single evaluation requires --kp, --ki, and --i-max")
    return args


def apply_cli_overrides(config: EvaluationConfig, args: argparse.Namespace) -> EvaluationConfig:
    overrides: dict[str, Any] = {}
    if args.output_root is not None:
        overrides["output_root"] = str(args.output_root)
    if args.reuse_policy is not None:
        overrides["reuse_policy"] = args.reuse_policy
    if args.mode is not None:
        overrides["mode"] = args.mode
    for raw in args.traction_reference:
        condition_id, value = parse_condition_assignment(raw, value_kind="float")
        overrides = deep_merge(
            overrides,
            nested_override(
                ["references", "traction_real_forces_n", condition_id], value
            ),
        )
    if args.frf_real_reference is not None:
        overrides = deep_merge(
            overrides,
            nested_override(
                ["references", "frf_real_reference"],
                str(args.frf_real_reference),
            ),
        )
    for raw in args.traction_sim_run:
        condition_id, value = parse_condition_assignment(raw, value_kind="path")
        overrides = deep_merge(
            overrides,
            nested_override(
                ["offline", "traction_sim_run_dirs", condition_id], value
            ),
        )
    if args.frf_sim_summary is not None:
        overrides = deep_merge(
            overrides,
            nested_override(["offline", "frf_sim_summary"], str(args.frf_sim_summary)),
        )
    return config.with_overrides(overrides)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = apply_cli_overrides(load_config(args.config), args)
    if config.mode not in {"offline", "managed"}:
        print(f"unsupported evaluation mode: {config.mode}", file=sys.stderr)
        return 2
    if config.mode == "managed" and args.no_runtime_assets:
        print("--no-runtime-assets cannot be used with --mode managed", file=sys.stderr)
        return 2

    if args.campaign_config is not None:
        spec = load_campaign_spec(args.campaign_config)
        if args.execute:
            campaign_output_dir = (
                config.output_root / "campaigns" / spec.campaign_id
            )
            try:
                summary = execute_managed_campaign(
                    spec,
                    config,
                    campaign_output_dir,
                    strict=bool(args.strict),
                    experiment_scope=args.experiment_scope,
                )
            except CampaignExecutionError as exc:
                print(f"campaign execution failed: {exc}", file=sys.stderr)
                return 1
            print(json.dumps(summary, indent=2, sort_keys=True))
            print(f"Campaign result: {campaign_output_dir / 'campaign_summary.json'}")
            return 0 if summary.get("scope_complete") is True else 1
        manifest = build_candidate_manifest(spec, config)
        state = build_initial_campaign_state(manifest)
        payload = {
            "campaign_config": str(args.campaign_config.resolve()),
            "campaign_manifest": manifest,
            "campaign_state_template": state,
            "cost_contract": frozen_cost_contract(),
            "experiment_scope": args.experiment_scope,
            "execution_authorized": False,
            "execution_requires": "--execute",
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    parameters = EvaluationParameters(kp=args.kp, ki=args.ki, i_max=args.i_max)
    parameters.validate()

    if args.dry_run:
        payload = {
            "parameters": parameters.to_dict(),
            "fixed_parameters": fixed_parameters_dict(),
            "cost_contract": frozen_cost_contract(),
            "parameter_label": parameters.label(),
            "parameter_fingerprint": parameters.fingerprint(config.to_dict()),
            "config": config.to_dict(),
        }
        if config.mode == "managed":
            payload["managed_plan"] = build_managed_dry_run(
                config,
                Path("<evaluation_output>"),
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if config.mode == "managed":
        result = evaluate_managed(
            parameters,
            config,
            create_assets=True,
            strict=bool(args.strict),
        )
    else:
        result = evaluate_offline(
            parameters,
            config,
            create_assets=not args.no_runtime_assets,
            strict=bool(args.strict),
        )
    print(json.dumps(result.to_dict()["cost"], indent=2, sort_keys=True))
    print(f"Result: {result.output_dir / 'evaluation_result.json'}")
    return 0 if result.cost.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
