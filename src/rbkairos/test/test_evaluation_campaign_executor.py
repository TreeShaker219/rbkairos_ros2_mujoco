from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

from evaluation.campaign import load_campaign_spec  # noqa: E402
from evaluation.campaign_executor import (  # noqa: E402
    CampaignCandidateExecution,
    CampaignExecutionError,
    atomic_write_json,
    execute_managed_campaign,
)
from evaluation.config import load_config  # noqa: E402
from evaluation.costs import CostBreakdown  # noqa: E402
from evaluation.evaluators import (  # noqa: E402
    FRF_COST_METHOD,
    FrfMetric,
    TractionMetric,
)
from evaluation.result import EvaluationResult  # noqa: E402
from evaluation.traction_conditions import TRACTION_CONDITION_IDS  # noqa: E402
from evaluation.traction_outputs import TRACTION_OUTPUT_FILENAMES  # noqa: E402
import evaluate_parameters  # noqa: E402


WHEELS = (
    "left_front_wheel_joint",
    "right_front_wheel_joint",
    "left_back_wheel_joint",
    "right_back_wheel_joint",
)


def write_smoke_campaign(path: Path) -> Path:
    payload = {
        "schema_version": 4,
        "campaign_id": "conditional_reuse_smoke",
        "formal_search_authorized": False,
        "expected_candidate_count": 2,
        "fixed_parameters": {
            "mu_low": 0.01,
            "mu_high": 0.7,
            "effort_limit_nm": 100.0,
            "kd": 0.0,
            "i_min_rule": "-i_max",
        },
        "cost": {"tendon_weight": 0.5, "frf_weight": 0.5},
        "candidates": [
            {"kp": 50.0, "ki": 1.0, "i_max": 1.0},
            {"kp": 50.0, "ki": 1.0, "i_max": 3.0},
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def frequency_evidence(frequency: float, *, i_clamp: bool = False) -> dict[str, Any]:
    return {
        "schema": "rbkairos_frf_pid_nonbinding_evidence_v1",
        "frequency_hz": frequency,
        "complete": True,
        "controller_rate_latched": True,
        "diagnostic_schema_version": 2,
        "message_samples": 20,
        "per_wheel": {
            wheel: {
                "control_update_count": 200,
                "max_abs_candidate_i_term_nm": 1.0005 if i_clamp else 0.5,
                "i_clamp_ever": i_clamp,
                "effort_clamp_ever": False,
            }
            for wheel in WHEELS
        },
    }


def make_result(
    candidate: dict[str, Any],
    output_dir: Path,
    *,
    frf_metric: FrfMetric,
    frf_reused_from: str | None = None,
    certificate: dict[str, Any] | None = None,
    raw_paths: list[str] | None = None,
) -> EvaluationResult:
    parameters = candidate["parameters"]
    traction = {
        condition_id: TractionMetric(
            F_traction_real=400.0,
            F_traction_sim=400.0,
            relative_error=0.0,
            J_tendon_point=0.0,
            valid=True,
            reject_reason="",
            run_dir=str(output_dir / "traction" / condition_id),
            summary_path=str(output_dir / "traction" / condition_id / "summary.json"),
            diagnostics={},
        )
        for condition_id in TRACTION_CONDITION_IDS
    }
    return EvaluationResult(
        parameters=evaluate_parameters.EvaluationParameters(**parameters),
        cost=CostBreakdown(
            J_tendon_points={condition_id: 0.0 for condition_id in TRACTION_CONDITION_IDS},
            J_tendon_x=0.0,
            J_tendon_y=0.0,
            J_tendon=0.0,
            J_frf=frf_metric.J_frf,
            J_total=0.5 * frf_metric.J_frf,
            tendon_weight=0.5,
            frf_weight=0.5,
            valid=True,
            reject_reason="",
        ),
        traction=traction,
        frf=frf_metric,
        provenance={},
        output_dir=output_dir,
        frf_reused_from=frf_reused_from,
        frf_nonbinding_certificate=certificate,
        frf_source_raw_data_paths=raw_paths or [],
    )


class FakeCandidateRunner:
    def __init__(
        self,
        *,
        first_probe_clamps: bool = False,
        fail_once: str | None = None,
        drop_evidence: bool = False,
        invalid_frf: bool = False,
        invalid_traction_task: str | None = None,
        invalid_traction_probe_index: int = 0,
    ):
        self.first_probe_clamps = first_probe_clamps
        self.fail_once = fail_once
        self.drop_evidence = drop_evidence
        self.invalid_frf = invalid_frf
        self.invalid_traction_task = invalid_traction_task
        self.invalid_traction_probe_index = invalid_traction_probe_index
        self.failed = False
        self.calls: list[dict[str, Any]] = []
        self.physical_tasks: list[tuple[str, str]] = []

    def __call__(
        self,
        *,
        candidate: dict[str, Any],
        config: Any,
        output_dir: Path,
        completed_task_outputs: dict[str, str],
        frf_reuse_bundle: dict[str, Any] | None,
        task_callback: Any,
        strict: bool,
        experiment_scope: str = "all",
    ) -> CampaignCandidateExecution:
        del config, strict
        candidate_id = candidate["candidate_id"]
        self.calls.append(
            {
                "candidate_id": candidate_id,
                "completed": dict(completed_task_outputs),
                "reuse": frf_reuse_bundle,
                "experiment_scope": experiment_scope,
            }
        )
        raw_paths: dict[str, str] = {}
        evidence: list[dict[str, Any]] = []
        if experiment_scope in {"all", "tendon"}:
            for task_id in candidate["tasks"]["traction"]:
                if task_id in completed_task_outputs:
                    continue
                task_callback(task_id, "started", {})
                self.physical_tasks.append((candidate_id, task_id))
                raw_dir = output_dir / "raw" / task_id
                raw_dir.mkdir(parents=True, exist_ok=True)
                invalid_traction = (
                    task_id == self.invalid_traction_task
                    and candidate["frf_reuse"]["probe_order_index"]
                    == self.invalid_traction_probe_index
                )
                task_callback(
                    task_id,
                    "finished",
                    {
                        "status": "invalid" if invalid_traction else "complete",
                        "output_dir": str(raw_dir.parent),
                        "raw_data_path": str(raw_dir),
                        "last_error": (
                            "synthetic_not_converged" if invalid_traction else None
                        ),
                    },
                )

        if frf_reuse_bundle is None:
            if experiment_scope in {"all", "frf"} or any(
                task_id.startswith("frf_") for task_id in completed_task_outputs
            ):
                for task_id, frequency in zip(
                    candidate["tasks"]["frf"],
                    config_frequencies(candidate),
                ):
                    if task_id in completed_task_outputs:
                        raw_paths[task_id] = completed_task_outputs[task_id]
                        evidence.append(
                            frequency_evidence(
                                frequency,
                                i_clamp=self.first_probe_clamps
                                and candidate["frf_reuse"]["probe_order_index"] == 0,
                            )
                        )
                        continue
                    if experiment_scope not in {"all", "frf"}:
                        continue
                    task_callback(task_id, "started", {})
                    self.physical_tasks.append((candidate_id, task_id))
                    if self.fail_once == task_id and not self.failed:
                        self.failed = True
                        task_callback(
                            task_id,
                            "finished",
                            {
                                "status": "failed",
                                "output_dir": None,
                                "raw_data_path": None,
                                "last_error": "synthetic failure",
                            },
                        )
                        raise CampaignExecutionError("synthetic failure")
                    raw_dir = output_dir / "raw" / task_id
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    raw_paths[task_id] = str(raw_dir)
                    task_callback(
                        task_id,
                        "finished",
                        {
                            "status": "complete",
                            "output_dir": str(raw_dir.parent),
                            "raw_data_path": str(raw_dir),
                            "last_error": None,
                        },
                    )
                    evidence.append(
                        frequency_evidence(
                            frequency,
                            i_clamp=self.first_probe_clamps
                            and candidate["frf_reuse"]["probe_order_index"] == 0,
                        )
                    )
            frf_metric = FrfMetric(
                J_frf=0.25,
                cost_method=FRF_COST_METHOD,
                real_replicate_count=3,
                real_reference_path="/real/reference.csv",
                matched_frequencies=config_frequencies(candidate),
                gain_db_real=[0.0],
                gain_db_real_sample_std=[1.0],
                gain_db_sim=[0.5],
                normalized_frequency_weights=[1.0],
                per_frequency_error=[],
                valid_frequency_count=len(config_frequencies(candidate)),
                rejected_frequency_count=0,
                valid=True,
                reject_reason="",
                analysis_dir=str(output_dir / "frf" / "analysis"),
                summary_path=str(output_dir / "frf" / "analysis" / "summary.csv"),
                diagnostics={},
            )
            reuse_source = None
            certificate = None
        else:
            frf_metric = frf_reuse_bundle["frf_metric"]
            raw_paths = dict(frf_reuse_bundle["source_raw_data_paths"])
            reuse_source = frf_reuse_bundle["source_candidate_id"]
            certificate = frf_reuse_bundle["certificate"]

        result = make_result(
            candidate,
            output_dir,
            frf_metric=frf_metric,
            frf_reused_from=reuse_source,
            certificate=certificate,
            raw_paths=list(raw_paths.values()),
        )
        if self.invalid_frf and frf_reuse_bundle is None:
            result.frf.valid = False
            result.frf.reject_reason = "synthetic_invalid_frf"
        if (
            self.invalid_traction_task is not None
            and candidate["frf_reuse"]["probe_order_index"]
            == self.invalid_traction_probe_index
        ):
            condition_id = self.invalid_traction_task.removeprefix("traction_")
            metric = result.traction[condition_id]
            metric.F_traction_sim = None
            metric.relative_error = None
            metric.J_tendon_point = None
            metric.valid = False
            metric.reject_reason = "synthetic_not_converged"
            result.cost.J_tendon_points[condition_id] = None
            result.cost.J_tendon_x = None
            result.cost.J_tendon_y = None
            result.cost.J_tendon = None
            result.cost.J_total = None
            result.cost.valid = False
            result.cost.reject_reason = (
                f"traction_{condition_id}:synthetic_not_converged"
            )
        return CampaignCandidateExecution(
            result=result,
            frf_frequency_evidence=[] if self.drop_evidence else evidence,
            frf_source_raw_data_paths=raw_paths,
        )


def config_frequencies(candidate: dict[str, Any]) -> list[float]:
    result: list[float] = []
    for task_id in candidate["tasks"]["frf"]:
        token = task_id.removeprefix("frf_").removesuffix("hz").replace("p", ".")
        result.append(float(token))
    return result


def smoke_config():
    return load_config().with_overrides(
        {
            "frf_experiment": {"frequencies": [1.0]},
            "analyzers": {"traction": {"plots": False}, "frf": {"plots": False, "html": False}},
        }
    )


def test_repository_smoke_configs_freeze_the_bounded_runtime_plan():
    config_path = (
        PACKAGE_ROOT / "config" / "evaluation" / "conditional_frf_reuse_smoke.yaml"
    )
    campaign_path = (
        PACKAGE_ROOT
        / "config"
        / "evaluation"
        / "campaign_conditional_frf_smoke.yaml"
    )
    config = load_config(config_path)
    spec = load_campaign_spec(campaign_path)

    manifest = evaluate_parameters.build_candidate_manifest(spec, config)
    bounds = manifest["frf_reuse_plan"]["physical_task_bounds"]
    assert config.mode == "managed"
    assert spec.formal_search_authorized is False
    assert manifest["candidate_count"] == 2
    assert len(manifest["frf_reuse_plan"]["groups"]) == 1
    assert manifest["frf_frequencies_hz"] == [1.0, 4.0]
    assert bounds == {
        "traction": 12,
        "frf_best_case": 2,
        "frf_worst_case": 4,
        "experiments_best_case": 14,
        "experiments_worst_case": 16,
    }


def test_execute_reuses_frf_and_second_invocation_is_a_noop(tmp_path):
    spec = load_campaign_spec(write_smoke_campaign(tmp_path / "campaign.yaml"))
    runner = FakeCandidateRunner()
    output_dir = tmp_path / "campaign-output"

    summary = execute_managed_campaign(
        spec,
        smoke_config(),
        output_dir,
        candidate_runner=runner,
    )
    assert summary["status"] == "complete"
    assert summary["candidate_count"] == 2
    assert summary["physical_task_count"] == 13
    assert summary["reused_frf_task_count"] == 1

    state = json.loads((output_dir / "campaign_state.json").read_text())
    ordered = summary["candidate_ids"]
    source_id, target_id = ordered
    for candidate_id in ordered:
        candidate_dir = output_dir / "candidates" / candidate_id
        assert all(
            (candidate_dir / filename).is_file()
            for filename in TRACTION_OUTPUT_FILENAMES
        )
    assert state["candidates"][source_id]["status"] == "complete"
    assert state["candidates"][target_id]["status"] == "complete"
    target_reuse = state["candidates"][target_id]["frf_reuse"]
    assert target_reuse["frf_reused_from"] == source_id
    assert target_reuse["nonbinding_certificate"]["proves_nonbinding"] is True
    target_frf_task = next(
        task for name, task in state["candidates"][target_id]["tasks"].items()
        if name.startswith("frf_")
    )
    assert target_frf_task["status"] == "reused"
    assert target_frf_task["raw_data_path"]
    assert target_frf_task["nonbinding_certificate_fingerprint"]

    target_result = json.loads(
        Path(state["candidates"][target_id]["result_path"]).read_text()
    )
    assert target_result["frf_reused_from"] == source_id
    assert target_result["frf_nonbinding_certificate"]["proves_nonbinding"] is True
    assert target_result["frf_source_raw_data_paths"] == [
        target_frf_task["raw_data_path"]
    ]
    assert target_result["cost"]["J_frf"] == pytest.approx(0.25)

    calls_before = len(runner.calls)
    tasks_before = list(runner.physical_tasks)
    resumed = execute_managed_campaign(
        spec,
        smoke_config(),
        output_dir,
        candidate_runner=runner,
    )
    assert resumed["status"] == "complete"
    assert resumed["resumed"] is True
    assert len(runner.calls) == calls_before
    assert runner.physical_tasks == tasks_before


def test_binding_probe_runs_next_i_max_instead_of_reusing(tmp_path):
    spec = load_campaign_spec(write_smoke_campaign(tmp_path / "campaign.yaml"))
    runner = FakeCandidateRunner(first_probe_clamps=True)
    summary = execute_managed_campaign(
        spec,
        smoke_config(),
        tmp_path / "campaign-output",
        candidate_runner=runner,
    )
    assert summary["status"] == "complete"
    assert summary["physical_task_count"] == 14
    assert summary["reused_frf_task_count"] == 0
    assert len([task for _, task in runner.physical_tasks if task.startswith("frf_")]) == 2


@pytest.mark.parametrize("first_scope", ["tendon", "frf"])
def test_experiment_scopes_can_run_separately_and_then_complete(
    tmp_path,
    first_scope,
):
    spec = load_campaign_spec(write_smoke_campaign(tmp_path / "campaign.yaml"))
    runner = FakeCandidateRunner()
    output_dir = tmp_path / "campaign-output"
    second_scope = "frf" if first_scope == "tendon" else "tendon"

    first = execute_managed_campaign(
        spec,
        smoke_config(),
        output_dir,
        experiment_scope=first_scope,
        candidate_runner=runner,
    )
    assert first["status"] == "pending"
    assert first["experiment_scope"] == first_scope
    assert first["scope_complete"] is True
    assert first["ranking_candidate_ids"] == []
    assert all(
        task_id.startswith("traction_" if first_scope == "tendon" else "frf_")
        for _, task_id in runner.physical_tasks
    )

    second = execute_managed_campaign(
        spec,
        smoke_config(),
        output_dir,
        experiment_scope=second_scope,
        candidate_runner=runner,
    )
    assert second["status"] == "complete"
    assert second["experiment_scope"] == second_scope
    assert second["scope_complete"] is True
    assert second["physical_task_count"] == 13
    assert second["reused_frf_task_count"] == 1
    assert len(second["ranking_candidate_ids"]) == 2

    state = json.loads((output_dir / "campaign_state.json").read_text())
    assert all(
        candidate["status"] == "complete"
        for candidate in state["candidates"].values()
    )


def test_failed_task_is_resumed_without_repeating_completed_tasks(tmp_path):
    spec = load_campaign_spec(write_smoke_campaign(tmp_path / "campaign.yaml"))
    task_id = "frf_1p000000000hz"
    runner = FakeCandidateRunner(fail_once=task_id)
    output_dir = tmp_path / "campaign-output"

    with pytest.raises(CampaignExecutionError, match="synthetic failure"):
        execute_managed_campaign(
            spec,
            smoke_config(),
            output_dir,
            candidate_runner=runner,
        )
    first_physical = list(runner.physical_tasks)
    assert len([task for _, task in first_physical if task.startswith("traction_")]) == 6

    summary = execute_managed_campaign(
        spec,
        smoke_config(),
        output_dir,
        candidate_runner=runner,
    )
    assert summary["status"] == "complete"
    source_id = summary["candidate_ids"][0]
    source_traction = [
        item for item in runner.physical_tasks
        if item[0] == source_id and item[1].startswith("traction_")
    ]
    assert len(source_traction) == 6
    assert runner.physical_tasks.count((source_id, task_id)) == 2


def test_invalid_tendon_candidate_is_recorded_and_campaign_continues(tmp_path):
    spec = load_campaign_spec(write_smoke_campaign(tmp_path / "campaign.yaml"))
    runner = FakeCandidateRunner(
        invalid_traction_task="traction_y_v0p01",
    )
    output_dir = tmp_path / "campaign-output"

    summary = execute_managed_campaign(
        spec,
        smoke_config(),
        output_dir,
        candidate_runner=runner,
    )

    source_id, target_id = summary["candidate_ids"]
    assert summary["status"] == "complete"
    assert summary["scope_complete"] is True
    assert summary["valid_candidate_count"] == 1
    assert summary["invalid_candidate_count"] == 1
    assert summary["invalid_candidate_ids"] == [source_id]
    assert summary["physical_task_count"] == 13
    assert summary["ranking_candidate_ids"] == [target_id]
    assert summary["best_candidate_id"] == target_id

    state = json.loads((output_dir / "campaign_state.json").read_text())
    assert state["candidates"][source_id]["status"] == "invalid"
    assert state["candidates"][source_id]["tasks"]["traction_y_v0p01"][
        "status"
    ] == "invalid"
    assert state["candidates"][target_id]["status"] == "complete"
    assert [call["candidate_id"] for call in runner.calls] == [source_id, target_id]

    source_result = json.loads(
        Path(state["candidates"][source_id]["result_path"]).read_text()
    )
    assert source_result["cost"]["valid"] is False
    assert source_result["cost"]["J_total"] is None
    with (output_dir / "campaign_summary.csv").open(encoding="utf-8") as handle:
        summary_csv = handle.read()
    assert "reject_reason" in summary_csv
    assert "synthetic_not_converged" in summary_csv


def test_invalid_tendon_output_is_resumed_without_repeating_it(tmp_path):
    spec = load_campaign_spec(write_smoke_campaign(tmp_path / "campaign.yaml"))
    task_id = "frf_1p000000000hz"
    invalid_task = "traction_y_v0p01"
    runner = FakeCandidateRunner(
        fail_once=task_id,
        invalid_traction_task=invalid_task,
    )
    output_dir = tmp_path / "campaign-output"

    with pytest.raises(CampaignExecutionError, match="synthetic failure"):
        execute_managed_campaign(
            spec,
            smoke_config(),
            output_dir,
            candidate_runner=runner,
        )

    summary = execute_managed_campaign(
        spec,
        smoke_config(),
        output_dir,
        candidate_runner=runner,
    )

    source_id = summary["candidate_ids"][0]
    assert summary["status"] == "complete"
    assert summary["invalid_candidate_ids"] == [source_id]
    assert summary["physical_task_count"] == 13
    assert runner.physical_tasks.count((source_id, invalid_task)) == 1
    assert runner.physical_tasks.count((source_id, task_id)) == 2


@pytest.mark.parametrize(
    ("runner", "message"),
    [
        (FakeCandidateRunner(drop_evidence=True), "evidence"),
        (FakeCandidateRunner(invalid_frf=True), "invalid"),
    ],
)
def test_missing_evidence_or_invalid_frf_never_reuses(
    tmp_path,
    runner,
    message,
):
    spec = load_campaign_spec(write_smoke_campaign(tmp_path / "campaign.yaml"))
    output_dir = tmp_path / "campaign-output"
    with pytest.raises(CampaignExecutionError, match=message):
        execute_managed_campaign(
            spec,
            smoke_config(),
            output_dir,
            candidate_runner=runner,
        )
    state = json.loads((output_dir / "campaign_state.json").read_text())
    target_id = list(state["candidates"])[1]
    target = state["candidates"][target_id]
    assert target["frf_reuse"]["frf_reused_from"] is None
    assert all(
        task["status"] != "reused"
        for task_id, task in target["tasks"].items()
        if task_id.startswith("frf_")
    )


def test_atomic_write_never_leaves_temporary_file(tmp_path):
    path = tmp_path / "state.json"
    atomic_write_json(path, {"revision": 1})
    atomic_write_json(path, {"revision": 2})
    assert json.loads(path.read_text()) == {"revision": 2}
    assert not list(tmp_path.glob(".state.json.*.tmp"))


def test_campaign_cli_defaults_to_dry_run_and_requires_explicit_execute(tmp_path):
    path = write_smoke_campaign(tmp_path / "campaign.yaml")
    default_args = evaluate_parameters.parse_args(["--campaign-config", str(path)])
    assert default_args.dry_run is True
    assert default_args.execute is False

    execute_args = evaluate_parameters.parse_args(
        [
            "--campaign-config",
            str(path),
            "--execute",
            "--experiment-scope",
            "tendon",
        ]
    )
    assert execute_args.execute is True
    assert execute_args.dry_run is False
    assert execute_args.experiment_scope == "tendon"

    with pytest.raises(SystemExit):
        evaluate_parameters.parse_args(
            ["--campaign-config", str(path), "--execute", "--dry-run"]
        )
    with pytest.raises(SystemExit):
        evaluate_parameters.parse_args(
            ["--kp", "50", "--ki", "1", "--i-max", "1", "--execute"]
        )
    with pytest.raises(SystemExit):
        evaluate_parameters.parse_args(
            [
                "--kp", "50", "--ki", "1", "--i-max", "1",
                "--experiment-scope", "frf",
            ]
        )


def test_campaign_cli_execute_calls_executor_only_with_explicit_flag(
    monkeypatch,
    tmp_path,
    capsys,
):
    path = write_smoke_campaign(tmp_path / "campaign.yaml")
    calls: list[tuple[Path, str]] = []

    def fake_execute(
        spec,
        config,
        output_dir,
        strict=False,
        experiment_scope="all",
    ):
        del spec, config, strict
        calls.append((output_dir, experiment_scope))
        return {
            "status": "pending",
            "scope_complete": True,
            "candidate_count": 2,
        }

    monkeypatch.setattr(
        evaluate_parameters,
        "execute_managed_campaign",
        fake_execute,
    )
    returncode = evaluate_parameters.main(
        [
            "--campaign-config",
            str(path),
            "--output-root",
            str(tmp_path / "data"),
            "--execute",
            "--experiment-scope",
            "frf",
        ]
    )
    assert returncode == 0
    assert calls == [
        (
            tmp_path / "data" / "campaigns" / "conditional_reuse_smoke",
            "frf",
        )
    ]
    assert "Campaign result:" in capsys.readouterr().out


def test_unapproved_large_campaign_cannot_execute(tmp_path):
    path = write_smoke_campaign(tmp_path / "campaign.yaml")
    payload = yaml.safe_load(path.read_text())
    payload["expected_candidate_count"] = 3
    payload["candidates"].append({"kp": 60.0, "ki": 1.0, "i_max": 1.0})
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    spec = load_campaign_spec(path)
    with pytest.raises(CampaignExecutionError, match="formal_search_authorized"):
        execute_managed_campaign(
            spec,
            smoke_config(),
            tmp_path / "campaign-output",
            candidate_runner=FakeCandidateRunner(),
        )
