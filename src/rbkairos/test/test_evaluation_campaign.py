import json
from pathlib import Path
import sys

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

from evaluation.campaign import (  # noqa: E402
    CAMPAIGN_STATE_SCHEMA,
    CANDIDATE_MANIFEST_SCHEMA,
    build_candidate_manifest,
    build_initial_campaign_state,
    load_campaign_spec,
    validate_campaign_state,
)
from evaluation.config import load_config  # noqa: E402
from evaluation.costs import frozen_cost_contract  # noqa: E402
from evaluation.traction_conditions import TRACTION_CONDITION_IDS  # noqa: E402
import evaluate_parameters  # noqa: E402


FIXED = {
    "mu_low": 0.01,
    "mu_high": 0.7,
    "effort_limit_nm": 100.0,
    "kd": 0.0,
    "i_min_rule": "-i_max",
}


def write_campaign(path, **overrides):
    payload = {
        "schema_version": 4,
        "campaign_id": "manual_test",
        "expected_candidate_count": 2,
        "fixed_parameters": FIXED,
        "cost": {"tendon_weight": 0.5, "frf_weight": 0.5},
        "candidates": [
            {"kp": 30, "ki": 0.5, "i_max": 1.0},
            {"kp": 50, "ki": 1.0, "i_max": 3.0},
        ],
    }
    payload.update(overrides)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_explicit_candidates_expand_to_frozen_unique_manifest(tmp_path):
    path = write_campaign(tmp_path / "candidates.yaml")
    config = load_config().with_overrides(
        {"frf_experiment": {"frequencies": [0.5, 1.0]}}
    )

    manifest = build_candidate_manifest(load_campaign_spec(path), config)
    repeated = build_candidate_manifest(load_campaign_spec(path), config)

    assert repeated == manifest
    assert manifest["schema"] == CANDIDATE_MANIFEST_SCHEMA
    assert manifest["frozen"] is True
    assert manifest["candidate_count"] == 2
    assert manifest["fixed_parameters"] == FIXED
    assert manifest["outer_weights_frozen"] is True
    assert manifest["cost_contract"] == frozen_cost_contract()
    assert [item["condition_id"] for item in manifest["traction_conditions"]] == list(
        TRACTION_CONDITION_IDS
    )
    assert [item["reference_force_n"] for item in manifest["traction_conditions"]] == [
        412.02,
        441.45,
        490.50,
        353.16,
        372.78,
        392.40,
    ]
    assert manifest["frf_frequencies_hz"] == [0.5, 1.0]
    assert [item["parameters"] for item in manifest["candidates"]] == [
        {"kp": 30.0, "ki": 0.5, "i_max": 1.0},
        {"kp": 50.0, "ki": 1.0, "i_max": 3.0},
    ]
    assert len({item["label"] for item in manifest["candidates"]}) == 2
    assert len({item["fingerprint"] for item in manifest["candidates"]}) == 2
    assert len({item["candidate_id"] for item in manifest["candidates"]}) == 2
    assert len(manifest["manifest_fingerprint"]) == 16

    first = manifest["candidates"][0]
    assert first["i_min"] == pytest.approx(-1.0)
    assert first["tasks"]["traction"] == [
        f"traction_{condition_id}" for condition_id in TRACTION_CONDITION_IDS
    ]
    assert first["tasks"]["frf"] == ["frf_0p500000000hz", "frf_1p000000000hz"]
    assert first["task_counts"] == {
        "traction": 6,
        "frf": 2,
        "experiments": 8,
        "cleanup_phases": 8,
        "managed_phases": 16,
    }
    assert manifest["total_task_counts"] == {
        "candidates": 2,
        "traction": 12,
        "frf": 4,
        "experiments": 16,
        "cleanup_phases": 16,
        "managed_phases": 32,
    }


def test_values_grid_uses_deterministic_cartesian_order(tmp_path):
    path = write_campaign(
        tmp_path / "grid.yaml",
        expected_candidate_count=8,
        candidates=None,
        grid={
            "kp": {"values": [30, 50]},
            "ki": {"values": [0.5, 1.0]},
            "i_max": {"values": [1.0, 3.0]},
        },
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload.pop("candidates")
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    manifest = build_candidate_manifest(load_campaign_spec(path), load_config())

    assert [item["parameters"] for item in manifest["candidates"]] == [
        {"kp": 30.0, "ki": 0.5, "i_max": 1.0},
        {"kp": 30.0, "ki": 0.5, "i_max": 3.0},
        {"kp": 30.0, "ki": 1.0, "i_max": 1.0},
        {"kp": 30.0, "ki": 1.0, "i_max": 3.0},
        {"kp": 50.0, "ki": 0.5, "i_max": 1.0},
        {"kp": 50.0, "ki": 0.5, "i_max": 3.0},
        {"kp": 50.0, "ki": 1.0, "i_max": 1.0},
        {"kp": 50.0, "ki": 1.0, "i_max": 3.0},
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"grid": {"kp": {"values": [1]}}}, "mutually exclusive"),
        ({"candidates": None}, "exactly one"),
        ({"expected_candidate_count": 3}, "candidate count"),
        ({"fixed_parameters": {**FIXED, "mu_high": 0.65}}, "fixed parameter"),
        (
            {
                "candidates": [
                    {"kp": 30, "ki": 0.5, "i_max": 1},
                    {"kp": 30, "ki": 0.5, "i_max": 1},
                ]
            },
            "duplicate candidate",
        ),
        (
            {
                "candidates": [
                    {"kp": 30.001, "ki": 0.5, "i_max": 1},
                    {"kp": 30.002, "ki": 0.5, "i_max": 1},
                ]
            },
            "label collision",
        ),
    ],
)
def test_campaign_contract_rejects_ambiguous_or_drifting_inputs(
    tmp_path, overrides, message
):
    path = write_campaign(tmp_path / "bad.yaml", **overrides)
    with pytest.raises(ValueError, match=message):
        build_candidate_manifest(load_campaign_spec(path), load_config())


@pytest.mark.parametrize(
    "candidate",
    [
        {"kp": 0, "ki": 1, "i_max": 1},
        {"kp": 50, "ki": -1, "i_max": 1},
        {"kp": 50, "ki": 1, "i_max": 0},
        {"kp": 50, "ki": 1, "i_max": 1, "mu": 0.7},
    ],
)
def test_campaign_reuses_evaluation_parameter_validation(tmp_path, candidate):
    path = write_campaign(
        tmp_path / "invalid.yaml",
        expected_candidate_count=1,
        candidates=[candidate],
    )
    with pytest.raises(ValueError):
        build_candidate_manifest(load_campaign_spec(path), load_config())


@pytest.mark.parametrize(
    "bad_cost",
    [
        {"tendon_weight": 0.5, "frf_weight": 2.0},
        {"tendon_weight": True, "frf_weight": 0.5},
        {"tendon_weight": 0.5},
        {"tendon_weight": 0.5, "frf_weight": 0.5, "extra": 1.0},
    ],
)
def test_campaign_requires_exact_frozen_outer_weights(tmp_path, bad_cost):
    path = write_campaign(tmp_path / "weights.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["cost"] = bad_cost
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="cost contract"):
        load_campaign_spec(path)


def test_evaluation_config_rejects_outer_weight_drift(tmp_path):
    path = tmp_path / "weights.yaml"
    path.write_text(
        yaml.safe_dump(
            {"cost": {"tendon_weight": 0.5, "frf_weight": 2.0}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frozen"):
        load_config(path)


def test_initial_resume_state_matches_manifest_and_rejects_drift(tmp_path):
    manifest = build_candidate_manifest(
        load_campaign_spec(write_campaign(tmp_path / "state.yaml")),
        load_config().with_overrides(
            {"frf_experiment": {"frequencies": [1.0]}}
        ),
    )

    state = build_initial_campaign_state(manifest)

    assert state["schema"] == CAMPAIGN_STATE_SCHEMA
    assert state["manifest_fingerprint"] == manifest["manifest_fingerprint"]
    assert state["status"] == "pending"
    assert state["revision"] == 0
    assert set(state["candidates"]) == {
        item["candidate_id"] for item in manifest["candidates"]
    }
    for candidate in state["candidates"].values():
        assert candidate["status"] == "pending"
        assert candidate["attempts"] == 0
        assert len(candidate["tasks"]) == 7
        assert all(task["status"] == "pending" for task in candidate["tasks"].values())
    validate_campaign_state(state, manifest)

    state["manifest_fingerprint"] = "stale"
    with pytest.raises(ValueError, match="fingerprint"):
        validate_campaign_state(state, manifest)


def test_campaign_cli_defaults_to_dry_run_and_never_executes_without_authorization(
    monkeypatch, tmp_path, capsys
):
    path = write_campaign(tmp_path / "cli.yaml")
    monkeypatch.setattr(
        evaluate_parameters,
        "evaluate_managed",
        lambda *args, **kwargs: pytest.fail("campaign dry-run started managed evaluation"),
    )
    monkeypatch.setattr(
        evaluate_parameters,
        "evaluate_offline",
        lambda *args, **kwargs: pytest.fail("campaign dry-run started offline evaluation"),
    )

    returncode = evaluate_parameters.main(
        ["--campaign-config", str(path), "--dry-run"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert returncode == 0
    assert payload["campaign_manifest"]["candidate_count"] == 2
    assert payload["campaign_manifest"]["total_task_counts"] == {
        "candidates": 2,
        "traction": 12,
        "frf": 14,
        "experiments": 26,
        "cleanup_phases": 26,
        "managed_phases": 52,
    }
    assert all(
        len(candidate["tasks"]["traction"]) == 6
        and len(candidate["tasks"]["frf"]) == 7
        for candidate in payload["campaign_manifest"]["candidates"]
    )
    assert payload["campaign_state_template"]["status"] == "pending"
    assert payload["execution_authorized"] is False
    assert payload["cost_contract"] == frozen_cost_contract()

    default_args = evaluate_parameters.parse_args(["--campaign-config", str(path)])
    assert default_args.dry_run is True
    assert default_args.execute is False
    with pytest.raises(SystemExit):
        evaluate_parameters.parse_args(
            [
                "--campaign-config", str(path), "--dry-run",
                "--kp", "50", "--ki", "1", "--i-max", "3",
            ]
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "config/evaluation/campaign_candidates.example.yaml",
        "config/evaluation/campaign_grid.example.yaml",
    ],
)
def test_example_campaign_configs_are_parseable_and_explicitly_nonformal(relative_path):
    path = PACKAGE_ROOT / relative_path
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["formal_search_authorized"] is False
    assert payload["cost"] == {"tendon_weight": 0.5, "frf_weight": 0.5}
    manifest = build_candidate_manifest(load_campaign_spec(path), load_config())
    assert manifest["candidate_count"] == payload["expected_candidate_count"]
