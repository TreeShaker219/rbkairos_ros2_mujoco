import json
import os
from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

from evaluation.config import load_config  # noqa: E402
from evaluation.costs import (  # noqa: E402
    FROZEN_FRF_WEIGHT,
    FROZEN_TENDON_WEIGHT,
    compute_multispeed_total_cost,
    frozen_cost_contract,
)
from evaluation.managed import build_managed_dry_run  # noqa: E402
from evaluation.evaluators import (  # noqa: E402
    FRF_COST_METHOD,
    FrfMetric,
    TractionMetric,
)
from evaluation.metadata_validation import RuntimeMetadataValidation  # noqa: E402
from evaluation.parameters import (  # noqa: E402
    FIXED_EFFORT_LIMIT_NM,
    FIXED_KD,
    FIXED_MU_HIGH,
    FIXED_MU_LOW,
    EvaluationParameters,
)
from evaluation.traction_conditions import (  # noqa: E402
    TRACTION_CONDITION_IDS,
    condition_axis,
    condition_speed_mps,
)
import evaluate_parameters  # noqa: E402
from evaluation import result as evaluation_result  # noqa: E402


EXPECTED_REFERENCES = {
    "x_v0p01": 412.02,
    "x_v0p05": 441.45,
    "x_v0p10": 490.50,
    "y_v0p01": 353.16,
    "y_v0p05": 372.78,
    "y_v0p10": 392.40,
}


def test_evaluation_cli_source_is_executable_for_ros2_run_install():
    script = PACKAGE_ROOT / "scripts" / "evaluate_parameters.py"
    assert os.access(script, os.X_OK)


def test_evaluation_parameters_atomically_replace_mu_with_i_max():
    parameters = EvaluationParameters(kp=50.0, ki=1.0, i_max=3.0)

    parameters.validate()
    assert parameters.to_dict() == {"kp": 50.0, "ki": 1.0, "i_max": 3.0}
    assert parameters.label() == "kp50_ki1_imax3"
    assert not hasattr(parameters, "mu")
    with pytest.raises(TypeError):
        EvaluationParameters(kp=50.0, ki=1.0, mu=0.7)
    with pytest.raises(ValueError, match="i_max"):
        EvaluationParameters(kp=50.0, ki=1.0, i_max=0.0).validate()

    assert FIXED_MU_LOW == pytest.approx(0.01)
    assert FIXED_MU_HIGH == pytest.approx(0.7)
    assert FIXED_EFFORT_LIMIT_NM == pytest.approx(100.0)
    assert FIXED_KD == pytest.approx(0.0)


def test_default_config_has_exact_six_equal_weight_tendon_conditions():
    config = load_config()

    assert tuple(config.traction_real_forces) == TRACTION_CONDITION_IDS
    assert config.traction_real_forces == EXPECTED_REFERENCES
    assert tuple(config.traction_experiments) == TRACTION_CONDITION_IDS
    assert tuple(config.traction_sim_run_dirs) == TRACTION_CONDITION_IDS
    assert config.tendon_weight == pytest.approx(0.5)
    assert config.frf_weight == pytest.approx(0.5)
    assert config.frf_real_reference == Path(
        "data_real_lab/frf_x_A0p10_three_run_reference.csv"
    )
    assert frozen_cost_contract() == {
        "schema": "rbkairos_evaluation_cost_contract_v2",
        "tendon_weight": 0.5,
        "frf_weight": 0.5,
        "formula": "J_total=0.5*J_tendon+0.5*J_frf",
    }

    for condition_id, experiment in config.traction_experiments.items():
        axis = condition_axis(condition_id)
        speed = condition_speed_mps(condition_id)
        assert experiment["axis"] == axis
        assert experiment[f"velocity_{axis}"] == pytest.approx(speed)
        assert experiment.get("velocity_y" if axis == "x" else "velocity_x") is None


def test_six_point_cost_and_frf_use_frozen_equal_outer_weights():
    point_costs = {
        condition_id: float(index + 1) / 100.0
        for index, condition_id in enumerate(TRACTION_CONDITION_IDS)
    }

    frozen = compute_multispeed_total_cost(
        point_costs,
        J_frf=0.25,
    )
    assert frozen.J_tendon_x == pytest.approx((0.01 + 0.02 + 0.03) / 3.0)
    assert frozen.J_tendon_y == pytest.approx((0.04 + 0.05 + 0.06) / 3.0)
    assert frozen.J_tendon == pytest.approx(sum(point_costs.values()) / 6.0)
    assert frozen.J_frf == pytest.approx(0.25)
    assert frozen.J_total == pytest.approx(
        0.5 * frozen.J_tendon + 0.5 * frozen.J_frf
    )
    assert frozen.tendon_weight == pytest.approx(FROZEN_TENDON_WEIGHT)
    assert frozen.frf_weight == pytest.approx(FROZEN_FRF_WEIGHT)
    assert frozen.valid is True


@pytest.mark.parametrize("reason", ["not_converged", "effort_clamp"])
def test_invalid_tendon_condition_cannot_enter_six_point_cost(reason):
    point_costs = {condition_id: 0.01 for condition_id in TRACTION_CONDITION_IDS}
    point_costs["x_v0p01"] = None

    result = compute_multispeed_total_cost(
        point_costs,
        J_frf=0.25,
        reject_reasons=[f"traction_x_v0p01:{reason}"],
    )

    assert result.J_tendon_points["x_v0p01"] is None
    assert result.J_tendon is None
    assert result.J_total is None
    assert result.valid is False
    assert reason in result.reject_reason


def test_managed_dry_run_contains_six_tendon_sessions_then_unchanged_frf():
    config = load_config().with_overrides(
        {"frf_experiment": {"frequencies": [1.0]}}
    )

    plan = build_managed_dry_run(config, Path("/candidate"))
    phases = [step["phase"] for step in plan["steps"]]
    expected = []
    for condition_id in TRACTION_CONDITION_IDS:
        expected.extend([f"traction_{condition_id}", f"cleanup_{condition_id}"])
    expected.extend(["frf_1p000000000hz", "cleanup_frf_1p000000000hz"])
    assert phases == expected

    for condition_id in TRACTION_CONDITION_IDS:
        step = next(
            item for item in plan["steps"]
            if item["phase"] == f"traction_{condition_id}"
        )
        command = step["recorder_command"]
        axis = condition_axis(condition_id)
        speed = condition_speed_mps(condition_id)
        index = command.index(f"--velocity-{axis}")
        assert float(command[index + 1]) == pytest.approx(speed)
    frf_launch = next(
        step["launch_command"]
        for step in plan["steps"]
        if step["phase"].startswith("frf_")
    )
    assert "traction_tendon_enabled:=false" in frf_launch


def test_cli_has_i_max_and_condition_overrides_but_rejects_weight_and_mu_aliases():
    args = evaluate_parameters.parse_args(
        [
            "--kp", "50",
            "--ki", "1",
            "--i-max", "3",
            "--traction-reference", "x_v0p01=410",
            "--traction-sim-run", "y_v0p10=/tmp/y_v0p10",
            "--frf-real-reference", "/tmp/real_frf_reference.csv",
            "--dry-run",
        ]
    )
    config = evaluate_parameters.apply_cli_overrides(load_config(), args)
    assert args.i_max == pytest.approx(3.0)
    assert config.traction_real_forces["x_v0p01"] == pytest.approx(410.0)
    assert config.traction_sim_run_dirs["y_v0p10"] == Path("/tmp/y_v0p10")
    assert config.frf_real_reference == Path("/tmp/real_frf_reference.csv")
    assert config.tendon_weight == pytest.approx(0.5)
    assert config.frf_weight == pytest.approx(0.5)

    with pytest.raises(SystemExit):
        evaluate_parameters.parse_args(
            ["--kp", "50", "--ki", "1", "--i-max", "3", "--mu", "0.7"]
        )
    with pytest.raises(SystemExit):
        evaluate_parameters.parse_args(
            [
                "--kp", "50", "--ki", "1", "--i-max", "3",
                "--tendon-weight", "2",
            ]
        )
    with pytest.raises(SystemExit):
        evaluate_parameters.parse_args(
            [
                "--kp", "50", "--ki", "1", "--i-max", "3",
                "--frf-real-summary", "/tmp/legacy.csv",
            ]
        )


def test_dry_run_serialization_has_no_mu_candidate_alias(capsys):
    returncode = evaluate_parameters.main(
        ["--kp", "50", "--ki", "1", "--i-max", "3", "--dry-run"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert returncode == 0
    assert payload["parameters"] == {"kp": 50.0, "ki": 1.0, "i_max": 3.0}
    assert "mu" not in payload["parameters"]
    assert payload["fixed_parameters"] == {
        "mu_low": 0.01,
        "mu_high": 0.7,
        "effort_limit_nm": 100.0,
        "kd": 0.0,
        "i_min_rule": "-i_max",
    }
    assert payload["cost_contract"] == frozen_cost_contract()


def test_offline_evaluation_aggregates_all_six_conditions(monkeypatch, tmp_path):
    run_dirs = {
        condition_id: tmp_path / condition_id
        for condition_id in TRACTION_CONDITION_IDS
    }
    for path in run_dirs.values():
        path.mkdir()
    config = load_config().with_overrides(
        {
            "mode": "offline",
            "offline": {
                "traction_sim_run_dirs": {
                    key: str(value) for key, value in run_dirs.items()
                },
                "frf_sim_summary": str(tmp_path / "sim_frf.csv"),
            },
            "references": {
                "frf_real_reference": str(tmp_path / "real_frf_reference.csv")
            },
        }
    )
    point_costs = {
        condition_id: float(index + 1) / 100.0
        for index, condition_id in enumerate(TRACTION_CONDITION_IDS)
    }

    def traction_metric(run_dir, real_force_n, axis, analyzer_config, strict):
        condition_id = Path(run_dir).name
        return TractionMetric(
            F_traction_real=float(real_force_n),
            F_traction_sim=float(real_force_n),
            relative_error=0.0,
            J_tendon_point=point_costs[condition_id],
            valid=True,
            reject_reason="",
            run_dir=str(run_dir),
            summary_path=str(Path(run_dir) / "traction_tendon_summary.json"),
            diagnostics={"axis": axis},
        )

    monkeypatch.setattr(evaluation_result, "_evaluate_traction_metric", traction_metric)
    monkeypatch.setattr(
        evaluation_result,
        "evaluate_frf_offline",
        lambda *args, **kwargs: FrfMetric(
            J_frf=0.25,
            cost_method=FRF_COST_METHOD,
            real_replicate_count=3,
            real_reference_path="/real/reference.csv",
            matched_frequencies=[1.0],
            gain_db_real=[0.0],
            gain_db_real_sample_std=[1.0],
            gain_db_sim=[0.0],
            normalized_frequency_weights=[1.0],
            per_frequency_error=[{"f_hz": 1.0, "error": 0.0}],
            valid_frequency_count=1,
            rejected_frequency_count=0,
            valid=True,
            reject_reason="",
            analysis_dir=str(tmp_path),
            summary_path=str(tmp_path / "sim_frf.csv"),
            diagnostics={},
        ),
    )
    monkeypatch.setattr(
        evaluation_result,
        "validate_runtime_metadata_multi",
        lambda *args, **kwargs: RuntimeMetadataValidation(True, [], []),
    )

    result = evaluation_result.evaluate_offline(
        EvaluationParameters(kp=50.0, ki=1.0, i_max=3.0),
        config,
        output_dir=tmp_path / "evaluation",
        create_assets=False,
    )

    assert tuple(result.traction) == TRACTION_CONDITION_IDS
    assert result.cost.J_tendon_points == point_costs
    assert result.cost.J_tendon == pytest.approx(sum(point_costs.values()) / 6.0)
    assert result.cost.J_frf == pytest.approx(0.25)
    assert result.cost.J_total == pytest.approx(
        0.5 * result.cost.J_tendon + 0.5 * result.cost.J_frf
    )
    assert result.cost.valid is True
    payload = json.loads(
        (result.output_dir / "evaluation_result.json").read_text(encoding="utf-8")
    )
    assert tuple(payload["traction"]) == TRACTION_CONDITION_IDS
    assert payload["parameters"] == {"kp": 50.0, "ki": 1.0, "i_max": 3.0}
    assert payload["cost_contract"] == frozen_cost_contract()
    assert payload["provenance"]["cost_contract"] == frozen_cost_contract()
