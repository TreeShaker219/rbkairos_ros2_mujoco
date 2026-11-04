import math
from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

from evaluation import costs, evaluators  # noqa: E402
import traction_tendon_analysis  # noqa: E402


def test_traction_cost_uses_steady_tow_force_not_peak(monkeypatch, tmp_path):
    summary = {
        "axis": "x",
        "measurement_method": "tendon_limit",
        "steady_tow_force_n": 400.0,
        "peak_tow_force_n": 999.0,
        "steady_tow_force_mean_n": 401.0,
        "steady_tow_force_std_n": 1.0,
        "steady_tow_force_cv": 0.0025,
        "tendon_active_ratio": 1.0,
        "max_constraint_extension_m": 0.0002,
        "steady_base_velocity_axis_mps": 0.0,
        "wheel_saturation_ratio": 0.4,
        "limit_mode": "not_effort_limited",
        "valid": True,
        "reject_reason": "",
        "warnings": [],
    }
    monkeypatch.setattr(
        traction_tendon_analysis,
        "analyze_traction_tendon_run",
        lambda *args, **kwargs: dict(summary),
    )

    metric = evaluators.evaluate_traction_offline(
        tmp_path,
        500.0,
        axis="x",
        write_summary=False,
        plots=False,
    )

    assert metric.F_traction_sim == 400.0
    assert metric.F_traction_sim != summary["peak_tow_force_n"]
    assert metric.J_tendon_point == pytest.approx(0.04)
    assert metric.valid is True
    assert not any("spring" in key.lower() for key in metric.to_dict())
    assert costs.compute_traction_cost(400.0, 500.0) == pytest.approx(0.04)


def test_evaluator_forwards_the_full_adaptive_steady_contract(monkeypatch, tmp_path):
    captured = {}

    def analyze(*args, **kwargs):
        captured.update(kwargs)
        return {
            "axis": "x",
            "measurement_method": "tendon_limit",
            "steady_tow_force_n": 412.02,
            "valid": True,
            "reject_reason": "",
        }

    monkeypatch.setattr(traction_tendon_analysis, "analyze_traction_tendon_run", analyze)

    evaluators.evaluate_traction_offline(
        tmp_path,
        412.02,
        axis="x",
        steady_window=2.0,
        steady_window_step=0.25,
        min_observation=2.0,
        required_stable_windows=4,
        max_wait=60.0,
        max_force_slope=0.5,
        max_i_term_slope=0.005,
        min_i_clamp_ratio=0.95,
    )

    assert captured["steady_window_s"] == pytest.approx(2.0)
    assert captured["steady_window_step_s"] == pytest.approx(0.25)
    assert captured["min_observation_s"] == pytest.approx(2.0)
    assert captured["required_stable_windows"] == 4
    assert captured["max_wait_s"] == pytest.approx(60.0)
    assert captured["max_force_slope_n_s"] == pytest.approx(0.5)
    assert captured["max_i_term_slope_nm_s"] == pytest.approx(0.005)
    assert captured["min_i_clamp_ratio"] == pytest.approx(0.95)


def test_traction_evaluator_propagates_invalid_run(monkeypatch, tmp_path):
    monkeypatch.setattr(
        traction_tendon_analysis,
        "analyze_traction_tendon_run",
        lambda *args, **kwargs: {
            "axis": "y",
            "measurement_method": "tendon_limit",
            "steady_tow_force_n": math.nan,
            "valid": False,
            "reject_reason": "force_cv=0.2>0.05",
        },
    )

    metric = evaluators.evaluate_traction_offline(tmp_path, 500.0, axis="y")

    assert metric.valid is False
    assert metric.J_tendon_point is None
    assert metric.reject_reason == "force_cv=0.2>0.05"
