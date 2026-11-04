from __future__ import annotations

import csv
from dataclasses import dataclass
import fcntl
import io
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

from evaluation.campaign import (
    CampaignSpec,
    build_candidate_manifest,
    build_initial_campaign_state,
    validate_campaign_state,
)
from evaluation.config import EvaluationConfig
from evaluation.evaluators import FRF_COST_METHOD, FrfMetric
from evaluation.frf_reuse import (
    apply_frf_probe_certificate,
    build_frf_nonbinding_certificate,
)
from evaluation.result import EvaluationResult


class CampaignExecutionError(RuntimeError):
    pass


@dataclass
class CampaignCandidateExecution:
    result: EvaluationResult
    frf_frequency_evidence: list[dict[str, Any]]
    frf_source_raw_data_paths: dict[str, str]


CandidateRunner = Callable[..., CampaignCandidateExecution]
EXPERIMENT_SCOPES = ("all", "tendon", "frf")
RESUMABLE_TASK_STATUSES = {"complete", "invalid"}
RESOLVED_TASK_STATUSES = {"complete", "invalid", "reused"}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2, sort_keys=True, allow_nan=False)
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="") as file_obj:
            file_obj.write(text)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignExecutionError(f"cannot load campaign file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CampaignExecutionError(f"campaign file root must be an object: {path}")
    return payload


def _frf_metric_from_payload(payload: dict[str, Any]) -> FrfMetric:
    frf = payload.get("frf")
    if not isinstance(frf, dict):
        raise CampaignExecutionError("FRF reuse source result has no FRF metric")
    try:
        metric = FrfMetric(**frf)
    except TypeError as exc:
        raise CampaignExecutionError("FRF reuse source metric schema mismatch") from exc
    if not metric.valid or metric.J_frf is None:
        raise CampaignExecutionError("FRF reuse source metric is invalid")
    if metric.cost_method != FRF_COST_METHOD:
        raise CampaignExecutionError("FRF reuse source cost method mismatch")
    return metric


def _validate_execution_scope(spec: CampaignSpec, manifest: dict[str, Any]) -> str:
    if spec.formal_search_authorized:
        return "formal"
    group_count = len(manifest["frf_reuse_plan"]["groups"])
    if manifest["candidate_count"] > 2 or group_count != 1:
        raise CampaignExecutionError(
            "formal_search_authorized=false permits only a one-(Kp,Ki), "
            "at-most-two-candidate runtime smoke"
        )
    return "smoke"


def _prepare_files(
    manifest: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    manifest_path = output_dir / "campaign_manifest.json"
    state_path = output_dir / "campaign_state.json"
    resumed = manifest_path.exists() or state_path.exists()

    if manifest_path.exists():
        existing_manifest = _load_json(manifest_path)
        if existing_manifest != manifest:
            raise CampaignExecutionError(
                "existing campaign manifest differs from the frozen request"
            )
    else:
        atomic_write_json(manifest_path, manifest)

    if state_path.exists():
        state = _load_json(state_path)
        validate_campaign_state(state, manifest)
    else:
        state = build_initial_campaign_state(manifest)
        atomic_write_json(state_path, state)
    return manifest, state, resumed


def _candidate_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(candidate["candidate_id"]): candidate
        for candidate in manifest["candidates"]
    }


def _ordered_candidate_ids(manifest: dict[str, Any]) -> list[str]:
    return [
        candidate_id
        for group in manifest["frf_reuse_plan"]["groups"]
        for candidate_id in group["probe_order_candidate_ids"]
    ]


def _resumable_task_outputs(candidate_state: dict[str, Any]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for task_id, task in candidate_state["tasks"].items():
        status = task["status"]
        if status not in RESUMABLE_TASK_STATUSES:
            continue
        raw_path = task.get("raw_data_path")
        if not raw_path:
            raise CampaignExecutionError(
                f"{status} campaign task lacks raw_data_path: {task_id}"
            )
        outputs[task_id] = str(raw_path)
    return outputs


def _task_ids_for_scope(candidate: dict[str, Any], experiment_scope: str) -> list[str]:
    if experiment_scope == "tendon":
        return list(candidate["tasks"]["traction"])
    if experiment_scope == "frf":
        return list(candidate["tasks"]["frf"])
    return list(candidate["tasks"]["traction"]) + list(candidate["tasks"]["frf"])


def _tasks_are_resolved(
    candidate: dict[str, Any],
    candidate_state: dict[str, Any],
    experiment_scope: str,
) -> bool:
    return all(
        candidate_state["tasks"][task_id]["status"] in RESOLVED_TASK_STATUSES
        for task_id in _task_ids_for_scope(candidate, experiment_scope)
    )


def _scope_has_invalid_task(
    candidate: dict[str, Any],
    candidate_state: dict[str, Any],
    experiment_scope: str,
) -> bool:
    return any(
        candidate_state["tasks"][task_id]["status"] == "invalid"
        for task_id in _task_ids_for_scope(candidate, experiment_scope)
    )


def _scope_is_complete(
    manifest: dict[str, Any],
    state: dict[str, Any],
    experiment_scope: str,
) -> bool:
    candidates = _candidate_map(manifest)
    return all(
        _tasks_are_resolved(
            candidates[candidate_id],
            state["candidates"][candidate_id],
            experiment_scope,
        )
        for candidate_id in _ordered_candidate_ids(manifest)
    )


def _recover_interrupted_state(state: dict[str, Any]) -> bool:
    changed = False
    for candidate in state["candidates"].values():
        if candidate["status"] == "running":
            candidate["status"] = "pending"
            changed = True
        for task in candidate["tasks"].values():
            if task["status"] == "running":
                task["status"] = "failed"
                task["last_error"] = "interrupted_before_atomic_completion"
                task["finished_at_epoch"] = time.time()
                changed = True
    if state["status"] in {"running", "blocked"}:
        state["status"] = "pending"
        changed = True
    return changed


def _campaign_counts(state: dict[str, Any]) -> tuple[int, int]:
    physical = 0
    reused = 0
    for candidate in state["candidates"].values():
        for task in candidate["tasks"].values():
            if task["status"] in {"complete", "invalid"}:
                physical += 1
            elif task["status"] == "reused":
                reused += 1
    return physical, reused


def _write_campaign_summary(
    output_dir: Path,
    manifest: dict[str, Any],
    state: dict[str, Any],
    *,
    resumed: bool,
    execution_scope: str,
    experiment_scope: str,
) -> dict[str, Any]:
    ordered_ids = _ordered_candidate_ids(manifest)
    records: list[dict[str, Any]] = []
    for candidate_id in ordered_ids:
        candidate = _candidate_map(manifest)[candidate_id]
        candidate_state = state["candidates"][candidate_id]
        result_path = candidate_state.get("result_path")
        result_payload = _load_json(Path(result_path)) if result_path else None
        records.append(
            {
                "candidate_id": candidate_id,
                "parameters": candidate["parameters"],
                "status": candidate_state["status"],
                "result_path": result_path,
                "frf_reused_from": candidate_state["frf_reuse"].get(
                    "frf_reused_from"
                ),
                "nonbinding_certificate_fingerprint": (
                    candidate_state["frf_reuse"].get("nonbinding_certificate") or {}
                ).get("certificate_fingerprint"),
                "cost": result_payload.get("cost") if result_payload else None,
            }
        )
    physical, reused_count = _campaign_counts(state)
    valid_records = [
        record for record in records
        if record["status"] == "complete"
        and isinstance(record.get("cost"), dict)
        and record["cost"].get("valid") is True
    ]
    ranking = sorted(
        valid_records,
        key=lambda record: float(record["cost"]["J_total"]),
    )
    invalid_records = [
        record for record in records
        if record["status"] == "invalid"
    ]
    summary = {
        "schema": "rbkairos_evaluation_campaign_summary_v2",
        "campaign_id": manifest["campaign_id"],
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "status": state["status"],
        "execution_scope": execution_scope,
        "experiment_scope": experiment_scope,
        "scope_complete": _scope_is_complete(
            manifest,
            state,
            experiment_scope,
        ),
        "resumed": resumed,
        "candidate_count": len(records),
        "candidate_ids": ordered_ids,
        "valid_candidate_count": len(valid_records),
        "invalid_candidate_count": len(invalid_records),
        "invalid_candidate_ids": [
            record["candidate_id"] for record in invalid_records
        ],
        "physical_task_count": physical,
        "reused_frf_task_count": reused_count,
        "candidates": records,
        "ranking_candidate_ids": [record["candidate_id"] for record in ranking],
        "best_candidate_id": ranking[0]["candidate_id"] if ranking else None,
    }
    atomic_write_json(output_dir / "campaign_summary.json", summary)

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=[
            "candidate_id",
            "kp",
            "ki",
            "i_max",
            "status",
            "frf_reused_from",
            "nonbinding_certificate_fingerprint",
            "J_tendon",
            "J_frf",
            "J_total",
            "valid",
            "reject_reason",
            "result_path",
        ],
    )
    writer.writeheader()
    for record in records:
        cost = record.get("cost") or {}
        writer.writerow(
            {
                "candidate_id": record["candidate_id"],
                **record["parameters"],
                "status": record["status"],
                "frf_reused_from": record["frf_reused_from"] or "",
                "nonbinding_certificate_fingerprint": (
                    record["nonbinding_certificate_fingerprint"] or ""
                ),
                "J_tendon": cost.get("J_tendon", ""),
                "J_frf": cost.get("J_frf", ""),
                "J_total": cost.get("J_total", ""),
                "valid": cost.get("valid", ""),
                "reject_reason": cost.get("reject_reason", ""),
                "result_path": record["result_path"] or "",
            }
        )
    _atomic_write_text(output_dir / "campaign_summary.csv", csv_buffer.getvalue())
    return summary


def execute_managed_campaign(
    spec: CampaignSpec,
    config: EvaluationConfig,
    output_dir: Path,
    *,
    strict: bool = False,
    experiment_scope: str = "all",
    candidate_runner: CandidateRunner | None = None,
) -> dict[str, Any]:
    if config.mode != "managed":
        raise CampaignExecutionError("campaign execution requires evaluation mode=managed")
    if experiment_scope not in EXPERIMENT_SCOPES:
        raise CampaignExecutionError(
            "campaign experiment_scope must be one of "
            + ", ".join(EXPERIMENT_SCOPES)
        )
    manifest = build_candidate_manifest(spec, config)
    execution_scope = _validate_execution_scope(spec, manifest)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / "campaign.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignExecutionError("campaign is already running") from exc

        manifest, state, resumed = _prepare_files(manifest, output_dir)
        state_path = output_dir / "campaign_state.json"
        if _recover_interrupted_state(state):
            state["revision"] += 1
            atomic_write_json(state_path, state)
        if state["status"] == "complete":
            return _write_campaign_summary(
                output_dir,
                manifest,
                state,
                resumed=True,
                execution_scope=execution_scope,
                experiment_scope=experiment_scope,
            )

        if candidate_runner is None:
            from evaluation.managed import run_managed_campaign_candidate

            candidate_runner = run_managed_campaign_candidate

        candidates = _candidate_map(manifest)
        state["status"] = "running"
        state["execution_scope"] = execution_scope
        state["execution_authorized_by_cli"] = True
        state["revision"] += 1
        atomic_write_json(state_path, state)

        def save_state() -> None:
            state["revision"] += 1
            state["updated_at_epoch"] = time.time()
            validate_campaign_state(state, manifest)
            atomic_write_json(state_path, state)

        for candidate_id in _ordered_candidate_ids(manifest):
            candidate = candidates[candidate_id]
            candidate_state = state["candidates"][candidate_id]
            selected_tasks_resolved = _tasks_are_resolved(
                candidate, candidate_state, experiment_scope
            )
            all_tasks_resolved = _tasks_are_resolved(
                candidate, candidate_state, "all"
            )
            if (
                candidate_state["status"] in {"complete", "invalid"}
                and selected_tasks_resolved
            ) or (
                selected_tasks_resolved and not all_tasks_resolved
            ):
                continue
            reuse_status = candidate_state["frf_reuse"]["status"]
            if (
                experiment_scope in {"all", "frf"}
                and reuse_status == "waiting_for_probe"
            ):
                raise CampaignExecutionError(
                    f"FRF probe ordering is unresolved for {candidate_id}"
                )

            completed_outputs = _resumable_task_outputs(candidate_state)
            reuse_bundle: dict[str, Any] | None = None
            if reuse_status == "reused_nonbinding":
                source_id = candidate_state["frf_reuse"]["frf_reused_from"]
                source_result_path = state["candidates"][source_id].get("result_path")
                if not source_result_path:
                    raise CampaignExecutionError(
                        f"FRF reuse source has no result: {source_id}"
                    )
                source_payload = _load_json(Path(source_result_path))
                certificate = candidate_state["frf_reuse"]["nonbinding_certificate"]
                reuse_bundle = {
                    "source_candidate_id": source_id,
                    "certificate": certificate,
                    "frf_metric": _frf_metric_from_payload(source_payload),
                    "source_raw_data_paths": dict(
                        certificate["source_raw_data_paths"]
                    ),
                }

            candidate_state["status"] = "running"
            candidate_state["attempts"] += 1
            save_state()

            def task_callback(
                task_id: str,
                event: str,
                payload: dict[str, Any],
            ) -> None:
                if task_id not in candidate_state["tasks"]:
                    raise CampaignExecutionError(
                        f"candidate runner reported unknown task: {task_id}"
                    )
                task = candidate_state["tasks"][task_id]
                if task["status"] in RESOLVED_TASK_STATUSES:
                    raise CampaignExecutionError(
                        f"candidate runner repeated terminal task: {task_id}"
                    )
                if event == "started":
                    task["status"] = "running"
                    task["attempts"] += 1
                    task["started_at_epoch"] = time.time()
                    task["finished_at_epoch"] = None
                    task["last_error"] = None
                elif event == "finished":
                    status = payload.get("status")
                    if status not in {"complete", "failed", "invalid"}:
                        raise CampaignExecutionError(
                            f"candidate runner returned invalid task status: {status}"
                        )
                    task.update(
                        {
                            "status": status,
                            "output_dir": payload.get("output_dir"),
                            "raw_data_path": payload.get("raw_data_path"),
                            "last_error": payload.get("last_error"),
                            "finished_at_epoch": time.time(),
                        }
                    )
                else:
                    raise CampaignExecutionError(
                        f"candidate runner returned unknown task event: {event}"
                    )
                save_state()

            try:
                execution = candidate_runner(
                    candidate=candidate,
                    config=config,
                    output_dir=output_dir / "candidates" / candidate_id,
                    completed_task_outputs=completed_outputs,
                    frf_reuse_bundle=reuse_bundle,
                    task_callback=task_callback,
                    strict=strict,
                    experiment_scope=experiment_scope,
                )
            except Exception as exc:
                candidate_state["status"] = "failed"
                state["status"] = "blocked"
                save_state()
                if isinstance(exc, CampaignExecutionError):
                    raise
                raise CampaignExecutionError(
                    f"candidate runner failed: {candidate_id}: {exc}"
                ) from exc

            result = execution.result
            if not _tasks_are_resolved(candidate, candidate_state, experiment_scope):
                candidate_state["status"] = "failed"
                state["status"] = "blocked"
                result.write_json()
                candidate_state["result_path"] = str(
                    result.output_dir / "evaluation_result.json"
                )
                save_state()
                raise CampaignExecutionError(
                    f"candidate {experiment_scope} tasks are incomplete or failed: "
                    f"{candidate_id}"
                )

            if experiment_scope in {"all", "frf"} and (
                result.frf is None or not result.frf.valid
            ):
                candidate_state["status"] = "invalid"
                state["status"] = "blocked"
                result.write_json()
                candidate_state["result_path"] = str(
                    result.output_dir / "evaluation_result.json"
                )
                save_state()
                raise CampaignExecutionError(
                    f"candidate FRF evaluation is invalid: {candidate_id}: "
                    + (
                        result.frf.reject_reason
                        if result.frf is not None
                        else "missing_frf"
                    )
                )

            if (
                experiment_scope in {"all", "frf"}
                and reuse_status == "probe_pending"
            ):
                certificate = build_frf_nonbinding_certificate(
                    candidate_id=candidate_id,
                    parameters=candidate["parameters"],
                    expected_frequencies=manifest["frf_frequencies_hz"],
                    frequency_evidence=execution.frf_frequency_evidence,
                    source_raw_data_paths=execution.frf_source_raw_data_paths,
                )
                infrastructure_reasons = set(certificate["reject_reasons"]) - {
                    "i_clamp_observed",
                    "effort_clamp_observed",
                }
                if infrastructure_reasons:
                    candidate_state["status"] = "failed"
                    state["status"] = "blocked"
                    save_state()
                    raise CampaignExecutionError(
                        "FRF nonbinding evidence is incomplete or inconsistent: "
                        + ",".join(sorted(infrastructure_reasons))
                    )
                apply_frf_probe_certificate(
                    state,
                    manifest,
                    candidate_id,
                    certificate,
                )
                result.frf_nonbinding_certificate = certificate
                result.frf_source_raw_data_paths = list(
                    certificate["source_raw_data_paths"].values()
                )
            elif reuse_bundle is not None:
                result.frf_reused_from = reuse_bundle["source_candidate_id"]
                result.frf_nonbinding_certificate = reuse_bundle["certificate"]
                result.frf_source_raw_data_paths = list(
                    reuse_bundle["source_raw_data_paths"].values()
                )

            result.write_json()
            candidate_state["result_path"] = str(
                result.output_dir / "evaluation_result.json"
            )
            if _scope_has_invalid_task(candidate, candidate_state, "all"):
                candidate_state["status"] = "invalid"
                save_state()
                continue
            if _tasks_are_resolved(candidate, candidate_state, "all"):
                if not result.cost.valid or result.frf is None or not result.frf.valid:
                    candidate_state["status"] = "invalid"
                    state["status"] = "blocked"
                    save_state()
                    raise CampaignExecutionError(
                        f"candidate evaluation is invalid: {candidate_id}: "
                        f"{result.cost.reject_reason or 'missing_or_invalid_frf'}"
                    )
                result.append_csv()
                candidate_state["status"] = "complete"
            else:
                candidate_state["status"] = "pending"
            save_state()

        state["status"] = (
            "complete"
            if _scope_is_complete(manifest, state, "all")
            else "pending"
        )
        save_state()
        return _write_campaign_summary(
            output_dir,
            manifest,
            state,
            resumed=resumed,
            execution_scope=execution_scope,
            experiment_scope=experiment_scope,
        )
