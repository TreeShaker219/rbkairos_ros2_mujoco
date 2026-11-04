from __future__ import annotations

import hashlib
import json
import math
from typing import Any


FRF_REUSE_POLICY_SCHEMA = "rbkairos_frf_conditional_reuse_policy_v1"
FRF_NONBINDING_CERTIFICATE_SCHEMA = "rbkairos_frf_nonbinding_certificate_v1"
FRF_PID_NONBINDING_EVIDENCE_SCHEMA = "rbkairos_frf_pid_nonbinding_evidence_v1"
PID_DIAGNOSTIC_SCHEMA_VERSION = 2
WHEEL_NAMES = (
    "left_front_wheel_joint",
    "right_front_wheel_joint",
    "left_back_wheel_joint",
    "right_back_wheel_joint",
)


def frf_task_id(frequency_hz: float) -> str:
    return f"frf_{float(frequency_hz):.9f}".replace(".", "p") + "hz"


def _canonical_hash(payload: Any, length: int = 16) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def build_frf_reuse_plan(
    candidates: list[dict[str, Any]],
    frf_task_ids: list[str],
    traction_task_count: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for candidate in candidates:
        parameters = candidate["parameters"]
        key = (float(parameters["kp"]), float(parameters["ki"]))
        grouped.setdefault(key, []).append(candidate)

    groups: list[dict[str, Any]] = []
    assignments: dict[str, dict[str, Any]] = {}
    for key in sorted(grouped):
        kp, ki = key
        ordered = sorted(
            grouped[key],
            key=lambda item: (
                float(item["parameters"]["i_max"]),
                str(item["candidate_id"]),
            ),
        )
        group_id = f"kpki_{_canonical_hash({'kp': kp, 'ki': ki}, 12)}"
        candidate_ids = [str(item["candidate_id"]) for item in ordered]
        groups.append(
            {
                "group_id": group_id,
                "kp": kp,
                "ki": ki,
                "probe_order_candidate_ids": candidate_ids,
                "probe_order_i_max_nm": [
                    float(item["parameters"]["i_max"]) for item in ordered
                ],
            }
        )
        for index, item in enumerate(ordered):
            assignments[str(item["candidate_id"])] = {
                "group_id": group_id,
                "probe_order_index": index,
                "planned_action": "probe" if index == 0 else "conditional_probe",
                "waits_for_candidate_id": candidate_ids[index - 1] if index else None,
            }

    candidate_count = len(candidates)
    frequency_count = len(frf_task_ids)
    frf_best = len(groups) * frequency_count
    frf_worst = candidate_count * frequency_count
    plan = {
        "schema": FRF_REUSE_POLICY_SCHEMA,
        "enabled": True,
        "grouping_keys": ["kp", "ki"],
        "probe_order": "ascending_i_max",
        "decision_rule": (
            "the first probe with complete controller-rate evidence, no I-clamp "
            "latch, and no effort-clamp latch is reused by all larger i_max candidates"
        ),
        "required_pid_diagnostic_schema_version": PID_DIAGNOSTIC_SCHEMA_VERSION,
        "required_evidence_schema": FRF_PID_NONBINDING_EVIDENCE_SCHEMA,
        "groups": groups,
        "physical_task_bounds": {
            "traction": int(traction_task_count),
            "frf_best_case": frf_best,
            "frf_worst_case": frf_worst,
            "experiments_best_case": int(traction_task_count) + frf_best,
            "experiments_worst_case": int(traction_task_count) + frf_worst,
        },
    }
    return plan, assignments


def build_frf_nonbinding_certificate(
    *,
    candidate_id: str,
    parameters: dict[str, Any],
    expected_frequencies: list[float],
    frequency_evidence: list[dict[str, Any]],
    source_raw_data_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    i_max = float(parameters["i_max"])
    reject_reasons: list[str] = []
    expected = sorted(float(value) for value in expected_frequencies)
    observed_values = [
        float(item.get("frequency_hz", math.nan)) for item in frequency_evidence
    ]
    observed = sorted(value for value in observed_values if math.isfinite(value))
    if observed != expected or len(observed) != len(set(observed)):
        reject_reasons.append("frequency_coverage_mismatch")

    expected_task_ids = {frf_task_id(frequency) for frequency in expected}
    normalized_source_paths = {
        str(task_id): str(path)
        for task_id, path in (source_raw_data_paths or {}).items()
        if str(path).strip()
    }
    if set(normalized_source_paths) != expected_task_ids:
        reject_reasons.append("source_raw_data_paths_mismatch")

    max_peak = 0.0
    any_i_clamp = False
    any_effort_clamp = False
    min_update_count: int | None = None
    for evidence in frequency_evidence:
        if evidence.get("schema") != FRF_PID_NONBINDING_EVIDENCE_SCHEMA:
            reject_reasons.append("evidence_schema_mismatch")
            continue
        if evidence.get("diagnostic_schema_version") != PID_DIAGNOSTIC_SCHEMA_VERSION:
            reject_reasons.append("diagnostic_schema_mismatch")
        if evidence.get("complete") is not True:
            reject_reasons.append("incomplete_evidence")
        if evidence.get("controller_rate_latched") is not True:
            reject_reasons.append("not_controller_rate_latched")
        samples = evidence.get("message_samples")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
            reject_reasons.append("no_evidence_samples")
        per_wheel = evidence.get("per_wheel")
        if not isinstance(per_wheel, dict) or set(per_wheel) != set(WHEEL_NAMES):
            reject_reasons.append("wheel_coverage_mismatch")
            continue
        for wheel in WHEEL_NAMES:
            record = per_wheel[wheel]
            if not isinstance(record, dict):
                reject_reasons.append("invalid_wheel_evidence")
                continue
            count = record.get("control_update_count")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                reject_reasons.append("invalid_control_update_count")
            else:
                min_update_count = count if min_update_count is None else min(
                    min_update_count, count
                )
            try:
                peak = float(record.get("max_abs_candidate_i_term_nm"))
            except (TypeError, ValueError):
                peak = math.nan
            if not math.isfinite(peak) or peak < 0.0:
                reject_reasons.append("invalid_candidate_i_term_peak")
            else:
                max_peak = max(max_peak, peak)
                tolerance = 64.0 * math.ulp(max(1.0, abs(i_max), abs(peak)))
                if (
                    peak > i_max + tolerance
                    and not bool(record.get("i_clamp_ever"))
                ):
                    reject_reasons.append(
                        "candidate_i_term_peak_exceeds_bound_without_latch"
                    )
            any_i_clamp = any_i_clamp or bool(record.get("i_clamp_ever"))
            any_effort_clamp = any_effort_clamp or bool(
                record.get("effort_clamp_ever")
            )

    if any_i_clamp:
        reject_reasons.append("i_clamp_observed")
    if any_effort_clamp:
        reject_reasons.append("effort_clamp_observed")
    reject_reasons = list(dict.fromkeys(reject_reasons))

    certificate: dict[str, Any] = {
        "schema": FRF_NONBINDING_CERTIFICATE_SCHEMA,
        "candidate_id": str(candidate_id),
        "parameters": {
            "kp": float(parameters["kp"]),
            "ki": float(parameters["ki"]),
            "i_max": i_max,
        },
        "expected_frequencies_hz": expected,
        "observed_frequencies_hz": observed,
        "source_raw_data_paths": normalized_source_paths,
        "controller_rate_latched": all(
            evidence.get("controller_rate_latched") is True
            for evidence in frequency_evidence
        ) and bool(frequency_evidence),
        "diagnostic_schema_version": PID_DIAGNOSTIC_SCHEMA_VERSION,
        "max_abs_candidate_i_term_nm": max_peak,
        "i_bound_usage_ratio": max_peak / i_max if i_max > 0.0 else None,
        "minimum_control_update_count": min_update_count,
        "any_i_clamp_ever": any_i_clamp,
        "any_effort_clamp_ever": any_effort_clamp,
        "proves_nonbinding": not reject_reasons,
        "allows_reuse_for_i_max_greater_or_equal_nm": (
            i_max if not reject_reasons else None
        ),
        "reject_reasons": reject_reasons,
    }
    certificate["certificate_fingerprint"] = _canonical_hash(certificate)
    return certificate


def _manifest_candidate_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(candidate["candidate_id"]): candidate
        for candidate in manifest.get("candidates", [])
    }


def apply_frf_probe_certificate(
    state: dict[str, Any],
    manifest: dict[str, Any],
    candidate_id: str,
    certificate: dict[str, Any],
) -> None:
    candidates = _manifest_candidate_map(manifest)
    if candidate_id not in candidates:
        raise ValueError(f"unknown FRF probe candidate: {candidate_id}")
    if certificate.get("schema") != FRF_NONBINDING_CERTIFICATE_SCHEMA:
        raise ValueError("FRF nonbinding certificate schema mismatch")
    if certificate.get("candidate_id") != candidate_id:
        raise ValueError("FRF nonbinding certificate candidate mismatch")
    certificate_payload = dict(certificate)
    certificate_fingerprint = certificate_payload.pop("certificate_fingerprint", None)
    if certificate_fingerprint != _canonical_hash(certificate_payload):
        raise ValueError("FRF nonbinding certificate fingerprint mismatch")
    expected_parameters = candidates[candidate_id]["parameters"]
    if certificate.get("parameters") != expected_parameters:
        raise ValueError("FRF nonbinding certificate parameters mismatch")
    if certificate.get("expected_frequencies_hz") != sorted(
        float(value) for value in manifest.get("frf_frequencies_hz", [])
    ):
        raise ValueError("FRF nonbinding certificate frequencies mismatch")

    assignment = candidates[candidate_id]["frf_reuse"]
    group_id = assignment["group_id"]
    group_plan = next(
        group
        for group in manifest["frf_reuse_plan"]["groups"]
        if group["group_id"] == group_id
    )
    group_state = state["frf_reuse_groups"][group_id]
    if group_state["next_probe_candidate_id"] != candidate_id:
        raise ValueError("FRF certificate is not for the next ascending i_max probe")

    ordered = group_plan["probe_order_candidate_ids"]
    index = ordered.index(candidate_id)
    candidate_state = state["candidates"][candidate_id]
    candidate_state["frf_reuse"].update(
        {
            "status": (
                "measured_source"
                if certificate.get("proves_nonbinding") is True
                else "measured_binding"
            ),
            "frf_reused_from": None,
            "nonbinding_certificate": certificate,
        }
    )
    for task_id, task in candidate_state["tasks"].items():
        if task_id.startswith("frf_"):
            task["status"] = "complete"

    if certificate.get("proves_nonbinding") is True:
        group_state.update(
            {
                "status": "certified_nonbinding",
                "next_probe_candidate_id": None,
                "nonbinding_source_candidate_id": candidate_id,
                "nonbinding_certificate": certificate,
            }
        )
        for reused_id in ordered[index + 1:]:
            reused_state = state["candidates"][reused_id]
            reused_state["frf_reuse"].update(
                {
                    "status": "reused_nonbinding",
                    "frf_reused_from": candidate_id,
                    "nonbinding_certificate": certificate,
                }
            )
            for task_id, task in reused_state["tasks"].items():
                if task_id.startswith("frf_"):
                    raw_data_path = certificate["source_raw_data_paths"].get(
                        task_id
                    )
                    task.update(
                        {
                            "status": "reused",
                            "frf_reused_from": candidate_id,
                            "nonbinding_certificate_fingerprint": certificate[
                                "certificate_fingerprint"
                            ],
                            "output_dir": raw_data_path,
                            "raw_data_path": raw_data_path,
                        }
                    )
    else:
        next_id = ordered[index + 1] if index + 1 < len(ordered) else None
        group_state.update(
            {
                "status": "probing" if next_id else "exhausted",
                "next_probe_candidate_id": next_id,
            }
        )
        if next_id is not None:
            state["candidates"][next_id]["frf_reuse"]["status"] = "probe_pending"
    state["revision"] = int(state.get("revision", 0)) + 1
