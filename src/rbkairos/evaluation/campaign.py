from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from evaluation.config import EvaluationConfig
from evaluation.costs import frozen_cost_contract, validate_frozen_cost_mapping
from evaluation.frf_reuse import (
    FRF_NONBINDING_CERTIFICATE_SCHEMA,
    build_frf_reuse_plan,
    frf_task_id,
)
from evaluation.parameters import EvaluationParameters, fixed_parameters_dict
from evaluation.traction_conditions import (
    TRACTION_CONDITION_IDS,
    condition_axis,
    condition_speed_mps,
)


CAMPAIGN_SCHEMA_VERSION = 4
CANDIDATE_MANIFEST_SCHEMA = "rbkairos_evaluation_candidate_manifest_v4"
CAMPAIGN_STATE_SCHEMA = "rbkairos_evaluation_campaign_state_v4"
CAMPAIGN_STATUS_VALUES = {"pending", "running", "complete", "blocked"}
CANDIDATE_STATUS_VALUES = {"pending", "running", "complete", "failed", "invalid"}
TASK_STATUS_VALUES = {
    "pending", "running", "complete", "failed", "invalid", "reused"
}
PARAMETER_KEYS = ("kp", "ki", "i_max")
CAMPAIGN_KEYS = {
    "schema_version",
    "campaign_id",
    "expected_candidate_count",
    "fixed_parameters",
    "cost",
    "formal_search_authorized",
    "candidates",
    "grid",
}
CAMPAIGN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class CampaignSpec:
    campaign_id: str
    expected_candidate_count: int
    fixed_parameters: dict[str, float | str]
    parameters: tuple[EvaluationParameters, ...]
    source_sha256: str
    source_path: Path
    formal_search_authorized: bool


def _canonical_hash(payload: Any, length: int | None = None) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return digest[:length] if length is not None else digest


def _read_campaign_yaml(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise ValueError(f"campaign config does not exist: {path}")
    raw_bytes = path.read_bytes()
    try:
        payload = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot parse campaign config: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("campaign config root must be a mapping")
    return payload, hashlib.sha256(raw_bytes).hexdigest()


def _parameter_from_mapping(raw: Any, location: str) -> EvaluationParameters:
    if not isinstance(raw, dict):
        raise ValueError(f"{location} must be a mapping")
    if set(raw) != set(PARAMETER_KEYS):
        missing = sorted(set(PARAMETER_KEYS) - set(raw))
        extra = sorted(set(raw) - set(PARAMETER_KEYS))
        raise ValueError(
            f"{location} must contain exactly Kp/Ki/i_max: "
            f"missing={missing}, extra={extra}"
        )
    if any(isinstance(raw[key], bool) for key in PARAMETER_KEYS):
        raise ValueError(f"{location} values must be numeric, not boolean")
    try:
        parameters = EvaluationParameters(
            kp=float(raw["kp"]),
            ki=float(raw["ki"]),
            i_max=float(raw["i_max"]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} values must be numeric") from exc
    parameters.validate()
    return parameters


def _expand_candidates(raw: Any) -> list[EvaluationParameters]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("candidates must be a non-empty list")
    return [
        _parameter_from_mapping(item, f"candidates[{index}]")
        for index, item in enumerate(raw)
    ]


def _grid_values(raw_grid: Any, parameter_name: str) -> list[float]:
    if not isinstance(raw_grid, dict) or set(raw_grid) != set(PARAMETER_KEYS):
        raise ValueError("grid must contain exactly kp, ki, and i_max")
    entry = raw_grid.get(parameter_name)
    if not isinstance(entry, dict) or set(entry) != {"values"}:
        raise ValueError(f"grid.{parameter_name} must contain only values")
    values = entry.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError(f"grid.{parameter_name}.values must be a non-empty list")
    converted: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            raise ValueError(
                f"grid.{parameter_name}.values[{index}] must not be boolean"
            )
        try:
            converted.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"grid.{parameter_name}.values[{index}] must be numeric"
            ) from exc
    if len(converted) != len(set(converted)):
        raise ValueError(f"grid.{parameter_name}.values contains duplicates")
    return converted


def _expand_grid(raw: Any) -> list[EvaluationParameters]:
    values = {
        parameter_name: _grid_values(raw, parameter_name)
        for parameter_name in PARAMETER_KEYS
    }
    return [
        _parameter_from_mapping(
            {"kp": kp, "ki": ki, "i_max": i_max},
            f"grid candidate ({kp}, {ki}, {i_max})",
        )
        for kp, ki, i_max in itertools.product(
            values["kp"], values["ki"], values["i_max"]
        )
    ]


def _validate_unique_parameters(parameters: list[EvaluationParameters]) -> None:
    seen: set[tuple[float, float, float]] = set()
    for item in parameters:
        key = (float(item.kp), float(item.ki), float(item.i_max))
        if key in seen:
            raise ValueError(f"duplicate candidate parameters: {key}")
        seen.add(key)


def load_campaign_spec(path: Path) -> CampaignSpec:
    path = Path(path)
    payload, source_sha256 = _read_campaign_yaml(path)
    unknown = sorted(set(payload) - CAMPAIGN_KEYS)
    if unknown:
        raise ValueError(f"unknown campaign config keys: {unknown}")
    if payload.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        raise ValueError(
            f"campaign schema_version must be {CAMPAIGN_SCHEMA_VERSION}"
        )

    campaign_id = str(payload.get("campaign_id", ""))
    if not CAMPAIGN_ID_PATTERN.fullmatch(campaign_id):
        raise ValueError(
            "campaign_id must use lowercase letters, digits, underscores, or hyphens"
        )
    expected_count = payload.get("expected_candidate_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
    ):
        raise ValueError("expected_candidate_count must be a positive integer")
    formal_search_authorized = payload.get("formal_search_authorized", False)
    if not isinstance(formal_search_authorized, bool):
        raise ValueError("formal_search_authorized must be boolean")

    expected_fixed = fixed_parameters_dict()
    actual_fixed = payload.get("fixed_parameters")
    if actual_fixed != expected_fixed:
        raise ValueError(
            "campaign fixed parameter contract mismatch: "
            f"expected={expected_fixed}, actual={actual_fixed}"
        )

    validate_frozen_cost_mapping(payload.get("cost"), label="campaign")

    has_candidates = payload.get("candidates") is not None
    has_grid = payload.get("grid") is not None
    if has_candidates and has_grid:
        raise ValueError("candidates and grid are mutually exclusive")
    if not has_candidates and not has_grid:
        raise ValueError("campaign must define exactly one of candidates or grid")
    parameters = (
        _expand_candidates(payload["candidates"])
        if has_candidates else _expand_grid(payload["grid"])
    )
    _validate_unique_parameters(parameters)
    if len(parameters) != expected_count:
        raise ValueError(
            "candidate count mismatch: "
            f"expected {expected_count}, expanded {len(parameters)}"
        )
    return CampaignSpec(
        campaign_id=campaign_id,
        expected_candidate_count=expected_count,
        fixed_parameters=dict(expected_fixed),
        parameters=tuple(parameters),
        source_sha256=source_sha256,
        source_path=path.resolve(),
        formal_search_authorized=formal_search_authorized,
    )


def _frf_contract(config: EvaluationConfig) -> tuple[list[float], list[str]]:
    raw = config.frf_experiment.get("frequencies")
    if not isinstance(raw, list) or not raw:
        raise ValueError("frf_experiment.frequencies must be a non-empty list")
    frequencies: list[float] = []
    for index, value in enumerate(raw):
        try:
            frequency = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"FRF frequency {index} is not numeric") from exc
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError(f"FRF frequency {index} must be finite and positive")
        frequencies.append(frequency)
    if len(frequencies) != len(set(frequencies)):
        raise ValueError("frf_experiment.frequencies contains duplicates")
    task_ids = [frf_task_id(frequency) for frequency in frequencies]
    return frequencies, task_ids


def _traction_contract(config: EvaluationConfig) -> list[dict[str, Any]]:
    references = config.traction_real_forces
    result: list[dict[str, Any]] = []
    for condition_id in TRACTION_CONDITION_IDS:
        reference = references[condition_id]
        if reference is None or not math.isfinite(reference) or reference <= 0.0:
            raise ValueError(
                f"traction reference must be finite and positive: {condition_id}"
            )
        result.append(
            {
                "condition_id": condition_id,
                "axis": condition_axis(condition_id),
                "speed_mps": condition_speed_mps(condition_id),
                "reference_force_n": float(reference),
            }
        )
    return result


def _task_counts(traction_count: int, frf_count: int) -> dict[str, int]:
    experiment_count = traction_count + frf_count
    return {
        "traction": traction_count,
        "frf": frf_count,
        "experiments": experiment_count,
        "cleanup_phases": experiment_count,
        "managed_phases": 2 * experiment_count,
    }


def build_candidate_manifest(
    spec: CampaignSpec,
    config: EvaluationConfig,
) -> dict[str, Any]:
    if config.mode != "managed":
        raise ValueError("campaign requires evaluation mode=managed")
    config.validate_cost_contract()
    if config.frf_experiment.get("require_pid_nonbinding_evidence") is not True:
        raise ValueError(
            "campaign conditional FRF reuse requires PID nonbinding evidence"
        )

    evaluation_context = config.to_dict()
    evaluation_context_fingerprint = _canonical_hash(evaluation_context, 16)
    traction_conditions = _traction_contract(config)
    traction_tasks = [
        f"traction_{condition_id}" for condition_id in TRACTION_CONDITION_IDS
    ]
    frf_frequencies, frf_tasks = _frf_contract(config)
    counts = _task_counts(len(traction_tasks), len(frf_tasks))
    candidates: list[dict[str, Any]] = []
    labels: set[str] = set()
    fingerprints: set[str] = set()
    candidate_ids: set[str] = set()

    for index, parameters in enumerate(spec.parameters):
        label = parameters.label()
        if label in labels:
            raise ValueError(f"candidate label collision: {label}")
        labels.add(label)
        fingerprint = parameters.fingerprint(evaluation_context)
        if fingerprint in fingerprints:
            raise ValueError(f"candidate fingerprint collision: {fingerprint}")
        fingerprints.add(fingerprint)
        candidate_id = f"{label}_{fingerprint}"
        if candidate_id in candidate_ids:
            raise ValueError(f"candidate ID collision: {candidate_id}")
        candidate_ids.add(candidate_id)
        candidates.append(
            {
                "index": index,
                "candidate_id": candidate_id,
                "parameters": parameters.to_dict(),
                "i_min": -float(parameters.i_max),
                "label": label,
                "fingerprint": fingerprint,
                "tasks": {
                    "traction": list(traction_tasks),
                    "frf": list(frf_tasks),
                },
                "task_counts": dict(counts),
            }
        )

    total_counts = {
        "candidates": len(candidates),
        **{
            key: value * len(candidates)
            for key, value in counts.items()
        },
    }
    frf_reuse_plan, frf_assignments = build_frf_reuse_plan(
        candidates,
        frf_tasks,
        traction_task_count=len(candidates) * len(traction_tasks),
    )
    for candidate in candidates:
        candidate["frf_reuse"] = frf_assignments[candidate["candidate_id"]]
    manifest: dict[str, Any] = {
        "schema": CANDIDATE_MANIFEST_SCHEMA,
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "frozen": True,
        "execution_authorized": bool(spec.formal_search_authorized),
        "formal_search_authorized": bool(spec.formal_search_authorized),
        "campaign_id": spec.campaign_id,
        "campaign_config_sha256": spec.source_sha256,
        "evaluation_config_fingerprint": evaluation_context_fingerprint,
        "fixed_parameters": dict(spec.fixed_parameters),
        "outer_weights_frozen": True,
        "cost_contract": frozen_cost_contract(),
        "traction_conditions": traction_conditions,
        "frf_frequencies_hz": frf_frequencies,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "total_task_counts": total_counts,
        "frf_reuse_plan": frf_reuse_plan,
    }
    manifest["manifest_fingerprint"] = _canonical_hash(manifest, 16)
    return manifest


def build_initial_campaign_state(manifest: dict[str, Any]) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for item in manifest.get("candidates", []):
        task_ids = list(item["tasks"]["traction"]) + list(item["tasks"]["frf"])
        probe_index = int(item["frf_reuse"]["probe_order_index"])
        candidates[item["candidate_id"]] = {
            "status": "pending",
            "attempts": 0,
            "tasks": {
                task_id: {
                    "status": "pending",
                    "attempts": 0,
                    "output_dir": None,
                    "raw_data_path": None,
                    "last_error": None,
                    "started_at_epoch": None,
                    "finished_at_epoch": None,
                    **(
                        {
                            "frf_reused_from": None,
                            "nonbinding_certificate_fingerprint": None,
                        }
                        if task_id.startswith("frf_")
                        else {}
                    ),
                }
                for task_id in task_ids
            },
            "frf_reuse": {
                "status": "probe_pending" if probe_index == 0 else "waiting_for_probe",
                "frf_reused_from": None,
                "nonbinding_certificate": None,
            },
            "result_path": None,
        }
    frf_reuse_groups = {
        group["group_id"]: {
            "status": "probing",
            "next_probe_candidate_id": group["probe_order_candidate_ids"][0],
            "nonbinding_source_candidate_id": None,
            "nonbinding_certificate": None,
        }
        for group in manifest["frf_reuse_plan"]["groups"]
    }
    return {
        "schema": CAMPAIGN_STATE_SCHEMA,
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": manifest.get("campaign_id"),
        "manifest_fingerprint": manifest.get("manifest_fingerprint"),
        "status": "pending",
        "revision": 0,
        "candidates": candidates,
        "frf_reuse_groups": frf_reuse_groups,
    }


def validate_campaign_state(
    state: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if not isinstance(state, dict):
        raise ValueError("campaign state must be a mapping")
    if state.get("schema") != CAMPAIGN_STATE_SCHEMA:
        raise ValueError("campaign state schema mismatch")
    if state.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        raise ValueError("campaign state schema_version mismatch")
    if state.get("manifest_fingerprint") != manifest.get("manifest_fingerprint"):
        raise ValueError("campaign state manifest fingerprint mismatch")
    if state.get("campaign_id") != manifest.get("campaign_id"):
        raise ValueError("campaign state campaign_id mismatch")
    if state.get("status") not in CAMPAIGN_STATUS_VALUES:
        raise ValueError("campaign state status is invalid")
    revision = state.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("campaign state revision must be a non-negative integer")

    manifest_candidates = {
        item["candidate_id"]: item for item in manifest.get("candidates", [])
    }
    expected_candidates = {
        item["candidate_id"]: set(item["tasks"]["traction"] + item["tasks"]["frf"])
        for item in manifest.get("candidates", [])
    }
    actual_candidates = state.get("candidates")
    if not isinstance(actual_candidates, dict) or set(actual_candidates) != set(
        expected_candidates
    ):
        raise ValueError("campaign state candidates do not match manifest")
    for candidate_id, expected_tasks in expected_candidates.items():
        candidate = actual_candidates[candidate_id]
        if not isinstance(candidate, dict):
            raise ValueError(f"campaign state candidate is invalid: {candidate_id}")
        if candidate.get("status") not in CANDIDATE_STATUS_VALUES:
            raise ValueError(f"campaign candidate status is invalid: {candidate_id}")
        attempts = candidate.get("attempts")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
            raise ValueError(f"campaign candidate attempts is invalid: {candidate_id}")
        tasks = candidate.get("tasks")
        if not isinstance(tasks, dict) or set(tasks) != expected_tasks:
            raise ValueError(f"campaign tasks do not match manifest: {candidate_id}")
        for task_id, task in tasks.items():
            if not isinstance(task, dict) or task.get("status") not in TASK_STATUS_VALUES:
                raise ValueError(f"campaign task status is invalid: {task_id}")
            task_attempts = task.get("attempts")
            if (
                isinstance(task_attempts, bool)
                or not isinstance(task_attempts, int)
                or task_attempts < 0
            ):
                raise ValueError(f"campaign task attempts is invalid: {task_id}")
            if task.get("status") == "reused":
                if not task.get("frf_reused_from") or not task.get(
                    "nonbinding_certificate_fingerprint"
                ) or not task.get("raw_data_path"):
                    raise ValueError(f"reused FRF task lacks provenance: {task_id}")
        reuse = candidate.get("frf_reuse")
        if not isinstance(reuse, dict) or reuse.get("status") not in {
            "probe_pending",
            "waiting_for_probe",
            "measured_binding",
            "measured_source",
            "reused_nonbinding",
        }:
            raise ValueError(f"campaign FRF reuse state is invalid: {candidate_id}")
        reuse_status = reuse["status"]
        certificate = reuse.get("nonbinding_certificate")
        source_id = reuse.get("frf_reused_from")
        if reuse_status in {"measured_source", "measured_binding", "reused_nonbinding"}:
            if not isinstance(certificate, dict) or certificate.get(
                "schema"
            ) != FRF_NONBINDING_CERTIFICATE_SCHEMA:
                raise ValueError(f"campaign FRF certificate is invalid: {candidate_id}")
            payload = dict(certificate)
            fingerprint = payload.pop("certificate_fingerprint", None)
            if fingerprint != _canonical_hash(payload, 16):
                raise ValueError(
                    f"campaign FRF certificate fingerprint is invalid: {candidate_id}"
                )
        if reuse_status == "measured_source":
            if certificate.get("proves_nonbinding") is not True or source_id is not None:
                raise ValueError(f"campaign FRF source provenance is invalid: {candidate_id}")
        elif reuse_status == "measured_binding":
            if certificate.get("proves_nonbinding") is not False or source_id is not None:
                raise ValueError(f"campaign binding FRF provenance is invalid: {candidate_id}")
        elif reuse_status == "reused_nonbinding":
            if source_id not in manifest_candidates or certificate.get(
                "proves_nonbinding"
            ) is not True:
                raise ValueError(f"campaign reused FRF provenance is invalid: {candidate_id}")
            source_manifest = manifest_candidates[source_id]
            target_manifest = manifest_candidates[candidate_id]
            if (
                source_manifest["frf_reuse"]["group_id"]
                != target_manifest["frf_reuse"]["group_id"]
                or float(source_manifest["parameters"]["i_max"])
                > float(target_manifest["parameters"]["i_max"])
                or certificate.get("candidate_id") != source_id
            ):
                raise ValueError(f"campaign reused FRF crosses its proof boundary: {candidate_id}")
            expected_fingerprint = certificate.get("certificate_fingerprint")
            for task_id, task in tasks.items():
                if not task_id.startswith("frf_"):
                    continue
                if (
                    task.get("status") != "reused"
                    or task.get("frf_reused_from") != source_id
                    or task.get("nonbinding_certificate_fingerprint")
                    != expected_fingerprint
                    or task.get("raw_data_path")
                    != certificate.get("source_raw_data_paths", {}).get(task_id)
                ):
                    raise ValueError(
                        f"campaign reused FRF task provenance is invalid: {candidate_id}:{task_id}"
                    )

    expected_group_ids = {
        group["group_id"] for group in manifest["frf_reuse_plan"]["groups"]
    }
    group_states = state.get("frf_reuse_groups")
    if not isinstance(group_states, dict) or set(group_states) != expected_group_ids:
        raise ValueError("campaign FRF reuse groups do not match manifest")
    for group_id, group_state in group_states.items():
        if not isinstance(group_state, dict) or group_state.get("status") not in {
            "probing", "certified_nonbinding", "exhausted"
        }:
            raise ValueError(f"campaign FRF reuse group state is invalid: {group_id}")
        group_plan = next(
            group
            for group in manifest["frf_reuse_plan"]["groups"]
            if group["group_id"] == group_id
        )
        next_probe = group_state.get("next_probe_candidate_id")
        if next_probe is not None and next_probe not in group_plan[
            "probe_order_candidate_ids"
        ]:
            raise ValueError(f"campaign FRF next probe is outside its group: {group_id}")
