from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

from evaluation.campaign import (  # noqa: E402
    build_candidate_manifest,
    build_initial_campaign_state,
    load_campaign_spec,
    validate_campaign_state,
)
from evaluation.config import load_config  # noqa: E402
from evaluation.costs import CostBreakdown  # noqa: E402
from evaluation.evaluators import FRF_COST_METHOD, FrfMetric  # noqa: E402
from evaluation.frf_reuse import (  # noqa: E402
    FRF_NONBINDING_CERTIFICATE_SCHEMA,
    apply_frf_probe_certificate,
    build_frf_nonbinding_certificate,
)
from evaluation.parameters import EvaluationParameters  # noqa: E402
from evaluation.result import EvaluationResult  # noqa: E402
import frequency_response_recorder as recorder  # noqa: E402


FIXED = {
    "mu_low": 0.01,
    "mu_high": 0.7,
    "effort_limit_nm": 100.0,
    "kd": 0.0,
    "i_min_rule": "-i_max",
}
WHEELS = tuple(recorder.WHEEL_DOF_NAMES)


def write_grid(path: Path) -> Path:
    payload = {
        "schema_version": 4,
        "campaign_id": "frf_reuse_test",
        "expected_candidate_count": 6,
        "fixed_parameters": FIXED,
        "cost": {"tendon_weight": 0.5, "frf_weight": 0.5},
        "formal_search_authorized": False,
        "grid": {
            "kp": {"values": [30, 50]},
            "ki": {"values": [1.0]},
            "i_max": {"values": [3.0, 0.5, 1.5]},
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def build_manifest(tmp_path: Path):
    config = load_config().with_overrides(
        {"frf_experiment": {"frequencies": [0.2, 1.0]}}
    )
    return build_candidate_manifest(load_campaign_spec(write_grid(tmp_path / "grid.yaml")), config)


def frequency_evidence(frequency: float, *, peak: float = 0.48,
                       i_clamp: bool = False, effort_clamp: bool = False):
    return {
        "schema": "rbkairos_frf_pid_nonbinding_evidence_v1",
        "frequency_hz": frequency,
        "complete": True,
        "controller_rate_latched": True,
        "diagnostic_schema_version": 2,
        "message_samples": 100,
        "per_wheel": {
            wheel: {
                "control_update_count": 1000,
                "max_abs_candidate_i_term_nm": peak,
                "i_clamp_ever": i_clamp,
                "effort_clamp_ever": effort_clamp,
            }
            for wheel in WHEELS
        },
    }


def source_paths(*frequencies: float):
    return {
        f"frf_{frequency:.9f}".replace(".", "p") + "hz": f"/raw/{frequency:g}"
        for frequency in frequencies
    }


def test_manifest_groups_by_kp_ki_and_orders_probes_by_increasing_i_max(tmp_path):
    manifest = build_manifest(tmp_path)

    groups = manifest["frf_reuse_plan"]["groups"]
    assert len(groups) == 2
    assert manifest["frf_reuse_plan"]["grouping_keys"] == ["kp", "ki"]
    assert manifest["frf_reuse_plan"]["probe_order"] == "ascending_i_max"
    for group in groups:
        candidates = [
            next(item for item in manifest["candidates"] if item["candidate_id"] == candidate_id)
            for candidate_id in group["probe_order_candidate_ids"]
        ]
        assert [item["parameters"]["i_max"] for item in candidates] == [0.5, 1.5, 3.0]
        assert candidates[0]["frf_reuse"]["planned_action"] == "probe"
        assert candidates[1]["frf_reuse"]["planned_action"] == "conditional_probe"

    bounds = manifest["frf_reuse_plan"]["physical_task_bounds"]
    assert bounds == {
        "traction": 36,
        "frf_best_case": 4,
        "frf_worst_case": 12,
        "experiments_best_case": 40,
        "experiments_worst_case": 48,
    }


def test_controller_rate_certificate_requires_every_frequency_and_never_clamped():
    certificate = build_frf_nonbinding_certificate(
        candidate_id="kp50_ki1_imax0p5_deadbeef",
        parameters={"kp": 50.0, "ki": 1.0, "i_max": 0.5},
        expected_frequencies=[0.2, 1.0],
        frequency_evidence=[frequency_evidence(0.2), frequency_evidence(1.0, peak=0.49)],
        source_raw_data_paths=source_paths(0.2, 1.0),
    )

    assert certificate["schema"] == FRF_NONBINDING_CERTIFICATE_SCHEMA
    assert certificate["proves_nonbinding"] is True
    assert certificate["max_abs_candidate_i_term_nm"] == pytest.approx(0.49)
    assert certificate["allows_reuse_for_i_max_greater_or_equal_nm"] == pytest.approx(0.5)
    assert certificate["any_i_clamp_ever"] is False
    assert certificate["any_effort_clamp_ever"] is False
    assert len(certificate["certificate_fingerprint"]) == 16

    missing = build_frf_nonbinding_certificate(
        candidate_id="candidate",
        parameters={"kp": 50.0, "ki": 1.0, "i_max": 0.5},
        expected_frequencies=[0.2, 1.0],
        frequency_evidence=[frequency_evidence(0.2)],
        source_raw_data_paths=source_paths(0.2),
    )
    assert missing["proves_nonbinding"] is False
    assert "frequency_coverage_mismatch" in missing["reject_reasons"]

    clamped = build_frf_nonbinding_certificate(
        candidate_id="candidate",
        parameters={"kp": 50.0, "ki": 1.0, "i_max": 0.5},
        expected_frequencies=[0.2, 1.0],
        frequency_evidence=[
            frequency_evidence(0.2, peak=0.5002, i_clamp=True),
            frequency_evidence(1.0),
        ],
        source_raw_data_paths=source_paths(0.2, 1.0),
    )
    assert clamped["proves_nonbinding"] is False
    assert "i_clamp_observed" in clamped["reject_reasons"]
    assert (
        "candidate_i_term_peak_exceeds_bound_without_latch"
        not in clamped["reject_reasons"]
    )

    inconsistent_evidence = frequency_evidence(0.2)
    inconsistent_evidence["per_wheel"][WHEELS[0]].update(
        {
            "max_abs_candidate_i_term_nm": 0.5002,
            "i_clamp_ever": False,
        }
    )
    inconsistent_evidence["per_wheel"][WHEELS[1]].update(
        {
            "max_abs_candidate_i_term_nm": 0.5002,
            "i_clamp_ever": True,
        }
    )
    inconsistent = build_frf_nonbinding_certificate(
        candidate_id="candidate",
        parameters={"kp": 50.0, "ki": 1.0, "i_max": 0.5},
        expected_frequencies=[0.2, 1.0],
        frequency_evidence=[inconsistent_evidence, frequency_evidence(1.0)],
        source_raw_data_paths=source_paths(0.2, 1.0),
    )
    assert inconsistent["proves_nonbinding"] is False
    assert "i_clamp_observed" in inconsistent["reject_reasons"]
    assert (
        "candidate_i_term_peak_exceeds_bound_without_latch"
        in inconsistent["reject_reasons"]
    )

    effort_limited = build_frf_nonbinding_certificate(
        candidate_id="candidate",
        parameters={"kp": 50.0, "ki": 1.0, "i_max": 0.5},
        expected_frequencies=[0.2, 1.0],
        frequency_evidence=[
            frequency_evidence(0.2, effort_clamp=True),
            frequency_evidence(1.0),
        ],
        source_raw_data_paths=source_paths(0.2, 1.0),
    )
    assert effort_limited["proves_nonbinding"] is False
    assert "effort_clamp_observed" in effort_limited["reject_reasons"]


def test_state_machine_reuses_larger_i_max_only_after_a_valid_certificate(tmp_path):
    manifest = build_manifest(tmp_path)
    state = build_initial_campaign_state(manifest)
    group = manifest["frf_reuse_plan"]["groups"][0]
    source_id, middle_id, largest_id = group["probe_order_candidate_ids"]
    source = next(item for item in manifest["candidates"] if item["candidate_id"] == source_id)
    certificate = build_frf_nonbinding_certificate(
        candidate_id=source_id,
        parameters=source["parameters"],
        expected_frequencies=[0.2, 1.0],
        frequency_evidence=[frequency_evidence(0.2), frequency_evidence(1.0)],
        source_raw_data_paths=source_paths(0.2, 1.0),
    )

    apply_frf_probe_certificate(state, manifest, source_id, certificate)
    validate_campaign_state(state, manifest)

    assert state["candidates"][source_id]["frf_reuse"]["status"] == "measured_source"
    for reused_id in (middle_id, largest_id):
        reuse = state["candidates"][reused_id]["frf_reuse"]
        assert reuse["status"] == "reused_nonbinding"
        assert reuse["frf_reused_from"] == source_id
        assert reuse["nonbinding_certificate"] == certificate
        assert all(
            task["status"] == "reused"
            for task_id, task in state["candidates"][reused_id]["tasks"].items()
            if task_id.startswith("frf_")
        )


def test_binding_probe_advances_to_next_i_max_without_reusing(tmp_path):
    manifest = build_manifest(tmp_path)
    state = build_initial_campaign_state(manifest)
    group = manifest["frf_reuse_plan"]["groups"][0]
    first_id, second_id, third_id = group["probe_order_candidate_ids"]
    first = next(item for item in manifest["candidates"] if item["candidate_id"] == first_id)
    certificate = build_frf_nonbinding_certificate(
        candidate_id=first_id,
        parameters=first["parameters"],
        expected_frequencies=[0.2, 1.0],
        frequency_evidence=[frequency_evidence(0.2, i_clamp=True), frequency_evidence(1.0)],
        source_raw_data_paths=source_paths(0.2, 1.0),
    )

    apply_frf_probe_certificate(state, manifest, first_id, certificate)
    validate_campaign_state(state, manifest)

    assert state["candidates"][first_id]["frf_reuse"]["status"] == "measured_binding"
    assert state["candidates"][second_id]["frf_reuse"]["status"] == "probe_pending"
    assert state["candidates"][third_id]["frf_reuse"]["status"] == "waiting_for_probe"
    assert all(
        state["candidates"][item]["frf_reuse"]["frf_reused_from"] is None
        for item in (first_id, second_id, third_id)
    )


def test_state_machine_rejects_tampered_or_wrong_frequency_certificate(tmp_path):
    manifest = build_manifest(tmp_path)
    state = build_initial_campaign_state(manifest)
    source_id = manifest["frf_reuse_plan"]["groups"][0][
        "probe_order_candidate_ids"
    ][0]
    source = next(item for item in manifest["candidates"] if item["candidate_id"] == source_id)
    certificate = build_frf_nonbinding_certificate(
        candidate_id=source_id,
        parameters=source["parameters"],
        expected_frequencies=[0.2, 1.0],
        frequency_evidence=[frequency_evidence(0.2), frequency_evidence(1.0)],
        source_raw_data_paths=source_paths(0.2, 1.0),
    )
    certificate["max_abs_candidate_i_term_nm"] = 0.1
    with pytest.raises(ValueError, match="fingerprint"):
        apply_frf_probe_certificate(state, manifest, source_id, certificate)

    certificate = build_frf_nonbinding_certificate(
        candidate_id=source_id,
        parameters=source["parameters"],
        expected_frequencies=[0.2],
        frequency_evidence=[frequency_evidence(0.2)],
        source_raw_data_paths=source_paths(0.2),
    )
    with pytest.raises(ValueError, match="frequencies"):
        apply_frf_probe_certificate(state, manifest, source_id, certificate)


def make_recorder_message(*, count: int, peak: float, i_ever: bool = False):
    names = list(WHEELS)
    return SimpleNamespace(
        velocity_pid_diagnostic_schema_version=2,
        velocity_pid_joint_names=names,
        velocity_pid_target=[1.0] * 4,
        velocity_pid_actual=[0.5] * 4,
        velocity_pid_error=[0.5] * 4,
        velocity_pid_p_term=[25.0] * 4,
        velocity_pid_i_term=[0.2] * 4,
        velocity_pid_candidate_i_term=[0.2] * 4,
        velocity_pid_peak_abs_candidate_i_term=[peak] * 4,
        velocity_pid_d_term=[0.0] * 4,
        velocity_pid_raw_output=[25.2] * 4,
        velocity_pid_effort_clamped_output=[25.2] * 4,
        velocity_pid_i_clamp_active=[False] * 4,
        velocity_pid_effort_clamp_active=[False] * 4,
        velocity_pid_i_clamp_ever=[i_ever] * 4,
        velocity_pid_effort_clamp_ever=[False] * 4,
        velocity_pid_control_update_count=[count] * 4,
        velocity_pid_kp=[50.0] * 4,
        velocity_pid_ki=[1.0] * 4,
        velocity_pid_kd=[0.0] * 4,
        velocity_pid_i_max=[0.5] * 4,
        velocity_pid_i_min=[-0.5] * 4,
        velocity_pid_antiwindup=[True] * 4,
        velocity_pid_effort_limit=[100.0] * 4,
        dof_names=["world_joint#0", *names],
        qfrc_applied=[0.0, *([25.2] * 4)],
    )


def test_recorder_aggregates_monotonic_controller_latches_without_raw_csv():
    evidence = recorder.PidNonbindingEvidence(frequency_hz=0.2)
    evidence.update(recorder.decode_velocity_pid_diagnostics(
        make_recorder_message(count=10, peak=0.4)
    ))
    evidence.update(recorder.decode_velocity_pid_diagnostics(
        make_recorder_message(count=20, peak=0.48)
    ))

    summary = evidence.to_dict()
    assert summary["complete"] is True
    assert summary["controller_rate_latched"] is True
    assert summary["message_samples"] == 2
    assert summary["per_wheel"][WHEELS[0]]["control_update_count"] == 20
    assert summary["per_wheel"][WHEELS[0]]["max_abs_candidate_i_term_nm"] == pytest.approx(0.48)

    with pytest.raises(recorder.PidDiagnosticContractError, match="decreased"):
        evidence.update(recorder.decode_velocity_pid_diagnostics(
            make_recorder_message(count=19, peak=0.48)
        ))


def test_result_schema_records_reuse_source_and_certificate(tmp_path):
    certificate = build_frf_nonbinding_certificate(
        candidate_id="source",
        parameters={"kp": 50.0, "ki": 1.0, "i_max": 0.5},
        expected_frequencies=[0.2],
        frequency_evidence=[frequency_evidence(0.2)],
        source_raw_data_paths=source_paths(0.2),
    )
    cost = CostBreakdown(
        J_tendon_points={}, J_tendon_x=0.1, J_tendon_y=0.1, J_tendon=0.1,
        J_frf=0.2, J_total=0.15, tendon_weight=0.5, frf_weight=0.5,
        valid=True, reject_reason="",
    )
    result = EvaluationResult(
        parameters=EvaluationParameters(kp=50.0, ki=1.0, i_max=1.5),
        cost=cost,
        traction=None,
        frf=FrfMetric(
            J_frf=0.2, cost_method=FRF_COST_METHOD, real_replicate_count=3,
            real_reference_path="/real/reference.csv",
            matched_frequencies=[0.2], gain_db_real=[0.0],
            gain_db_real_sample_std=[1.0], gain_db_sim=[0.0],
            normalized_frequency_weights=[1.0], per_frequency_error=[], valid_frequency_count=1,
            rejected_frequency_count=0, valid=True, reject_reason="",
            analysis_dir="/frf", summary_path="/frf/summary.csv", diagnostics={},
        ),
        provenance={},
        output_dir=tmp_path,
        frf_reused_from="source",
        frf_nonbinding_certificate=certificate,
        frf_source_raw_data_paths=list(source_paths(0.2).values()),
    )

    payload = result.to_dict()
    assert payload["frf_reused_from"] == "source"
    assert payload["frf_nonbinding_certificate"] == certificate
    assert payload["frf_source_raw_data_paths"] == ["/raw/0.2"]
    result.append_csv()
    header = (tmp_path / "evaluation_result.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "frf_reused_from" in header
    assert "frf_nonbinding_certificate_json" in header
    assert "frf_source_raw_data_paths_json" in header


def test_formal_frf_command_requires_nonbinding_evidence():
    from evaluation.runners import build_frf_command_from_config

    command = build_frf_command_from_config(
        {"frequencies": [0.2], "require_pid_nonbinding_evidence": True},
        Path("/tmp/frf"),
    )
    assert "--require-pid-nonbinding-evidence" in command
