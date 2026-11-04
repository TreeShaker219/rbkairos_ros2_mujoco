import csv
import json
from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

from evaluation.config import load_config  # noqa: E402
from evaluation.costs import CostBreakdown, frozen_cost_contract  # noqa: E402
from evaluation.evaluators import (  # noqa: E402
    FRF_COST_METHOD,
    FrfMetric,
    TractionMetric,
)
from evaluation.parameters import EvaluationParameters  # noqa: E402
from evaluation.result import EvaluationResult  # noqa: E402
from evaluation.traction_conditions import TRACTION_CONDITION_IDS  # noqa: E402
import evaluate_parameters  # noqa: E402


def metric(axis):
    return TractionMetric(
        F_traction_real=500.0,
        F_traction_sim=450.0,
        relative_error=-0.1,
        J_tendon_point=0.01,
        valid=True,
        reject_reason="",
        run_dir=f"/{axis}",
        summary_path=f"/{axis}/traction_tendon_summary.json",
        diagnostics={"axis": axis},
    )


def test_result_json_and_csv_have_only_canonical_traction_names(tmp_path):
    cost = CostBreakdown(
        J_tendon_points={item: 0.01 for item in TRACTION_CONDITION_IDS},
        J_tendon_x=0.01,
        J_tendon_y=0.01,
        J_tendon=0.01,
        J_frf=0.25,
        J_total=0.13,
        tendon_weight=0.5,
        frf_weight=0.5,
        valid=True,
        reject_reason="",
    )
    result = EvaluationResult(
        parameters=EvaluationParameters(kp=50.0, ki=0.1, i_max=3.0),
        cost=cost,
        traction={item: metric(item) for item in TRACTION_CONDITION_IDS},
        frf=FrfMetric(
            J_frf=0.25,
            cost_method=FRF_COST_METHOD,
            real_replicate_count=3,
            real_reference_path="/real/reference.csv",
            matched_frequencies=[1.0],
            gain_db_real=[0.0],
            gain_db_real_sample_std=[1.0],
            gain_db_sim=[0.0],
            normalized_frequency_weights=[1.0],
            per_frequency_error=[],
            valid_frequency_count=1,
            rejected_frequency_count=0,
            valid=True,
            reject_reason="",
            analysis_dir="/frf",
            summary_path="/frf/summary.csv",
            diagnostics={},
        ),
        provenance={},
        output_dir=tmp_path,
    )

    json_path = result.write_json()
    csv_path = result.append_csv()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    header = next(csv.reader(csv_path.open(encoding="utf-8")))

    assert set(payload) >= {
        "schema",
        "schema_version",
        "parameters",
        "fixed_parameters",
        "cost_contract",
        "cost",
        "traction",
        "frf",
        "provenance",
    }
    assert payload["schema"] == "rbkairos_evaluation_result_v5"
    assert payload["schema_version"] == 5
    assert payload["frf"]["cost_method"] == FRF_COST_METHOD
    assert payload["frf_reused_from"] is None
    assert payload["frf_nonbinding_certificate"] is None
    assert payload["frf_source_raw_data_paths"] == []
    assert payload["cost_contract"] == frozen_cost_contract()
    assert payload["provenance"]["cost_contract"] == frozen_cost_contract()
    serialized = json.dumps(payload)
    assert "spring" not in serialized.lower()
    assert {"J_tendon_x", "J_tendon_y", "J_tendon", "J_frf", "J_total"} <= set(
        payload["cost"]
    )
    assert "J_traction" not in serialized
    assert "J_FRF" not in serialized
    assert "J_tendon" in header
    assert "J_frf" in header
    assert "J_traction" not in header
    assert "J_FRF" not in header
    assert "i_max" in header
    assert "mu" not in header
    assert not any("spring" in field.lower() for field in header)


def test_config_and_cli_reject_spring_aliases(tmp_path):
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text("references:\n  spring_real_force_x_n: 412.02\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown evaluation config key"):
        load_config(legacy)
    with pytest.raises(SystemExit):
        evaluate_parameters.parse_args(
            [
                "--kp", "50", "--ki", "0.1", "--i-max", "3",
                "--spring-real-force-x", "1",
            ]
        )
    with pytest.raises(SystemExit):
        evaluate_parameters.parse_args(
            ["--kp", "50", "--ki", "0.1", "--i-max", "3", "--mu", "0.7"]
        )

    args = evaluate_parameters.parse_args(
        [
            "--kp",
            "50",
            "--ki",
            "0.1",
            "--i-max", "3",
            "--traction-reference", "x_v0p01=500",
            "--traction-sim-run", "x_v0p01=/tmp/x",
        ]
    )
    config = evaluate_parameters.apply_cli_overrides(load_config(), args)
    assert config.traction_real_forces["x_v0p01"] == 500.0
    assert config.traction_sim_run_dirs["x_v0p01"] == Path("/tmp/x")
