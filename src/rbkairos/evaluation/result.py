from __future__ import annotations

import csv
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.config import EvaluationConfig
from evaluation.costs import (
    CostBreakdown,
    compute_multispeed_total_cost,
    frozen_cost_contract,
)
from evaluation.evaluators import (
    FrfMetric,
    TractionMetric,
    evaluate_frf_offline,
    evaluate_traction_offline,
)
from evaluation.metadata_validation import validate_runtime_metadata_multi
from evaluation.parameters import EvaluationParameters, fixed_parameters_dict
from evaluation.runtime_assets import RuntimeAssets, create_runtime_assets
from evaluation.traction_conditions import (
    TRACTION_CONDITION_IDS,
    condition_axis,
)


EVALUATION_RESULT_SCHEMA = "rbkairos_evaluation_result_v5"
EVALUATION_RESULT_SCHEMA_VERSION = 5


def _git_commit_or_none(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            return None
        return value
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _create_runtime_assets_from_config(
    parameters: EvaluationParameters,
    config: EvaluationConfig,
    asset_dir: Path,
) -> RuntimeAssets:
    asset_config = config.runtime_assets
    source_urdf = Path(str(asset_config.get("source_urdf", "src/rbkairos/urdf/rbkairos.urdf")))
    source_mjcf = Path(str(asset_config.get("source_mjcf", "src/rbkairos/mujoco/rbkairos.xml")))
    source_config = Path(
        str(asset_config.get("source_config", "src/rbkairos/config/mujoco_mecanum_drive.yaml"))
    )
    raw_scenes = asset_config.get("source_traction_scenes", {})
    source_traction_scenes = {
        axis: Path(str(raw_scenes.get(axis)))
        for axis in ("x", "y")
        if isinstance(raw_scenes, dict) and raw_scenes.get(axis)
    }
    return create_runtime_assets(
        parameters,
        asset_dir,
        source_urdf=source_urdf,
        source_mjcf=source_mjcf,
        source_traction_scenes=source_traction_scenes,
        source_config=source_config,
    )


def _write_result(
    parameters: EvaluationParameters,
    cost: CostBreakdown,
    traction: TractionMetric | dict[str, TractionMetric] | None,
    frf: FrfMetric | None,
    provenance: dict[str, Any],
    output_dir: Path,
    frf_reused_from: str | None = None,
    frf_nonbinding_certificate: dict[str, Any] | None = None,
    frf_source_raw_data_paths: list[str] | None = None,
) -> EvaluationResult:
    result = EvaluationResult(
        parameters=parameters,
        cost=cost,
        traction=traction,
        frf=frf,
        provenance=provenance,
        output_dir=output_dir,
        frf_reused_from=frf_reused_from,
        frf_nonbinding_certificate=frf_nonbinding_certificate,
        frf_source_raw_data_paths=list(frf_source_raw_data_paths or []),
    )
    result.write_json()
    result.append_csv()
    return result


@dataclass
class EvaluationResult:
    parameters: EvaluationParameters
    cost: CostBreakdown
    traction: TractionMetric | dict[str, TractionMetric] | None
    frf: FrfMetric | None
    provenance: dict[str, Any]
    output_dir: Path
    frf_reused_from: str | None = None
    frf_nonbinding_certificate: dict[str, Any] | None = None
    frf_source_raw_data_paths: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        provenance = dict(self.provenance)
        provenance["cost_contract"] = frozen_cost_contract()
        return _json_ready(
            {
                "schema": EVALUATION_RESULT_SCHEMA,
                "schema_version": EVALUATION_RESULT_SCHEMA_VERSION,
                "parameters": self.parameters.to_dict(),
                "fixed_parameters": fixed_parameters_dict(),
                "cost_contract": frozen_cost_contract(),
                "cost": self.cost.to_dict(),
                "traction": (
                    {axis: metric.to_dict() for axis, metric in self.traction.items()}
                    if isinstance(self.traction, dict)
                    else self.traction.to_dict() if self.traction else None
                ),
                "frf": self.frf.to_dict() if self.frf else None,
                "frf_reused_from": self.frf_reused_from,
                "frf_nonbinding_certificate": self.frf_nonbinding_certificate,
                "frf_source_raw_data_paths": list(
                    self.frf_source_raw_data_paths or []
                ),
                "provenance": provenance,
                "output_dir": self.output_dir,
            }
        )

    def write_json(self) -> Path:
        path = self.output_dir / "evaluation_result.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_obj:
            json.dump(self.to_dict(), file_obj, indent=2, sort_keys=True, allow_nan=False)
            file_obj.write("\n")
        self.write_traction_outputs()
        self.write_frf_outputs()
        return path

    def write_traction_outputs(self) -> tuple[Path, ...]:
        from evaluation.traction_outputs import write_traction_candidate_outputs

        return write_traction_candidate_outputs(
            self.parameters,
            self.traction,
            self.output_dir,
        )

    def write_frf_outputs(self) -> tuple[Path, ...]:
        from evaluation.frf_outputs import write_frf_candidate_outputs

        return write_frf_candidate_outputs(
            self.parameters,
            self.frf,
            self.output_dir,
            frf_reused_from=self.frf_reused_from,
            frf_source_raw_data_paths=self.frf_source_raw_data_paths,
        )

    def append_csv(self) -> Path:
        path = self.output_dir / "evaluation_result.csv"
        fields = [
            "kp",
            "ki",
            "i_max",
            "i_min",
            "mu_low",
            "mu_high",
            "kd",
            "effort_limit_nm",
            *[f"J_tendon_{item}" for item in TRACTION_CONDITION_IDS],
            "J_tendon_x",
            "J_tendon_y",
            "J_tendon",
            "J_frf",
            "J_total",
            "tendon_weight",
            "frf_weight",
            "valid",
            "reject_reason",
            "traction_run_dirs_json",
            "frf_summary_path",
            "frf_reused_from",
            "frf_nonbinding_certificate_json",
            "frf_source_raw_data_paths_json",
        ]
        fixed = fixed_parameters_dict()
        traction_metrics = self.traction if isinstance(self.traction, dict) else {}
        row = {
            "kp": self.parameters.kp,
            "ki": self.parameters.ki,
            "i_max": self.parameters.i_max,
            "i_min": -self.parameters.i_max,
            "mu_low": fixed["mu_low"],
            "mu_high": fixed["mu_high"],
            "kd": fixed["kd"],
            "effort_limit_nm": fixed["effort_limit_nm"],
            **{
                f"J_tendon_{condition_id}": self.cost.J_tendon_points.get(
                    condition_id
                )
                for condition_id in TRACTION_CONDITION_IDS
            },
            "J_tendon_x": self.cost.J_tendon_x,
            "J_tendon_y": self.cost.J_tendon_y,
            "J_tendon": self.cost.J_tendon,
            "J_frf": self.cost.J_frf,
            "J_total": self.cost.J_total,
            "tendon_weight": self.cost.tendon_weight,
            "frf_weight": self.cost.frf_weight,
            "valid": self.cost.valid,
            "reject_reason": self.cost.reject_reason,
            "traction_run_dirs_json": json.dumps(
                {
                    condition_id: traction_metrics[condition_id].run_dir
                    for condition_id in TRACTION_CONDITION_IDS
                    if condition_id in traction_metrics
                },
                sort_keys=True,
            ),
            "frf_summary_path": self.frf.summary_path if self.frf else "",
            "frf_reused_from": self.frf_reused_from or "",
            "frf_nonbinding_certificate_json": json.dumps(
                self.frf_nonbinding_certificate,
                sort_keys=True,
            ) if self.frf_nonbinding_certificate is not None else "",
            "frf_source_raw_data_paths_json": json.dumps(
                list(self.frf_source_raw_data_paths or []),
                sort_keys=True,
            ),
        }
        exists = path.exists()
        with path.open("a", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        return path


def _make_output_dir(config: EvaluationConfig, parameters: EvaluationParameters) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    context_payload = {
        "config": config.to_dict(),
        "git_commit": _git_commit_or_none(Path.cwd()),
    }
    fingerprint = parameters.fingerprint(context_payload)
    return config.output_root / f"{timestamp}_{parameters.label()}_{fingerprint}"


def _evaluate_traction_metric(
    run_dir: Path,
    real_force_n: float,
    axis: str,
    analyzer_config: dict[str, Any],
    strict: bool,
) -> TractionMetric:
    return evaluate_traction_offline(
        run_dir,
        float(real_force_n),
        axis=axis,
        steady_window=float(analyzer_config.get("steady_window", 2.0)),
        steady_window_step=float(analyzer_config.get("steady_window_step", 0.25)),
        min_observation=float(analyzer_config.get("min_observation", 2.0)),
        required_stable_windows=int(
            analyzer_config.get("required_stable_windows", 4)
        ),
        max_wait=float(analyzer_config.get("max_wait", 60.0)),
        min_tendon_active_ratio=float(
            analyzer_config.get("min_tendon_active_ratio", 0.95)
        ),
        max_force_cv=float(analyzer_config.get("max_force_cv", 0.05)),
        max_force_slope=float(analyzer_config.get("max_force_slope_n_s", 0.5)),
        max_constraint_extension_m=float(
            analyzer_config.get("max_constraint_extension_m", 0.005)
        ),
        max_steady_base_velocity_mps=float(
            analyzer_config.get("max_steady_base_velocity_mps", 0.005)
        ),
        max_i_term_slope=float(
            analyzer_config.get("max_i_term_slope_nm_s", 0.005)
        ),
        min_i_clamp_ratio=float(analyzer_config.get("min_i_clamp_ratio", 0.95)),
        reject_on_wheel_saturation=bool(
            analyzer_config.get("reject_on_wheel_saturation", False)
        ),
        wheel_saturation_ratio_limit=float(
            analyzer_config.get("wheel_saturation_ratio_limit", 0.99)
        ),
        strict=bool(strict or analyzer_config.get("strict", False)),
        write_summary=bool(analyzer_config.get("write_summary", True)),
        plots=bool(analyzer_config.get("plots", False)),
    )


def evaluate_offline(
    parameters: EvaluationParameters,
    config: EvaluationConfig,
    output_dir: Path | None = None,
    create_assets: bool = True,
    strict: bool = False,
) -> EvaluationResult:
    started = time.time()
    parameters.validate()
    output_dir = output_dir or _make_output_dir(config, parameters)
    output_dir.mkdir(parents=True, exist_ok=True)

    assets: RuntimeAssets | None = None
    if create_assets:
        assets = _create_runtime_assets_from_config(
            parameters,
            config,
            output_dir / "runtime_assets",
        )

    traction_metrics: dict[str, TractionMetric] = {}
    frf_metric: FrfMetric | None = None
    reject_reasons: list[str] = []

    traction_cfg = config.traction_analyzer
    traction_run_dirs = config.traction_sim_run_dirs
    traction_real_forces = config.traction_real_forces
    for condition_id in TRACTION_CONDITION_IDS:
        axis = condition_axis(condition_id)
        run_dir = traction_run_dirs[condition_id]
        real_force = traction_real_forces[condition_id]
        if run_dir is None:
            reject_reasons.append(
                f"missing_offline.traction_sim_run_dirs.{condition_id}"
            )
            continue
        if real_force is None:
            reject_reasons.append(
                f"missing_references.traction_real_forces_n.{condition_id}"
            )
            continue
        traction_metrics[condition_id] = _evaluate_traction_metric(
            run_dir,
            float(real_force),
            axis,
            traction_cfg,
            strict=bool(strict),
        )
        if not traction_metrics[condition_id].valid:
            reject_reasons.append(
                f"traction_{condition_id}:"
                f"{traction_metrics[condition_id].reject_reason}"
            )

    if config.frf_sim_summary is None:
        reject_reasons.append("missing_offline.frf_sim_summary")
    elif config.frf_real_reference is None:
        reject_reasons.append("missing_references.frf_real_reference")
    else:
        frf_metric = evaluate_frf_offline(
            config.frf_sim_summary,
            config.frf_real_reference,
        )
        if not frf_metric.valid:
            reject_reasons.append(f"frf:{frf_metric.reject_reason}")

    metadata_validation = validate_runtime_metadata_multi(
        parameters,
        traction_run_dirs=traction_run_dirs,
        frf_sim_summary=config.frf_sim_summary,
    )
    reject_reasons.extend(metadata_validation.reject_reasons)

    cost = compute_multispeed_total_cost(
        {
            condition_id: (
                traction_metrics[condition_id].J_tendon_point
                if condition_id in traction_metrics else None
            )
            for condition_id in TRACTION_CONDITION_IDS
        },
        frf_metric.J_frf if frf_metric else None,
        reject_reasons=reject_reasons,
    )

    provenance = {
        "mode": config.mode,
        "git_commit": _git_commit_or_none(Path.cwd()),
        "config_snapshot": config.to_dict(),
        "runtime_assets": assets.to_dict() if assets else None,
        "cache_decision": {
            "reuse_policy": config.reuse_policy,
            "used_cache": False,
            "reason": "fresh_offline_evaluation",
        },
        "wall_time": {
            "started_at_epoch": started,
            "duration_s": time.time() - started,
        },
        "metadata_paths": {
            "traction_run_metadata": {
                condition_id: (
                    str(run_dir / "run_metadata.json") if run_dir else None
                )
                for condition_id, run_dir in traction_run_dirs.items()
            },
            "frf_sim_summary": str(config.frf_sim_summary) if config.frf_sim_summary else None,
            "frf_real_reference": (
                str(config.frf_real_reference)
                if config.frf_real_reference else None
            ),
        },
        "runtime_metadata_validation": metadata_validation.to_dict(),
        "launch_commands": [],
        "analyzer_commands": [],
    }

    result = EvaluationResult(
        parameters=parameters,
        cost=cost,
        traction=traction_metrics,
        frf=frf_metric,
        provenance=provenance,
        output_dir=output_dir,
    )
    result.write_json()
    result.append_csv()
    return result
