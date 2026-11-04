import csv
import hashlib
from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from evaluation.evaluators import evaluate_frf_offline  # noqa: E402
from evaluation.managed import build_managed_dry_run  # noqa: E402
from evaluation.config import load_config  # noqa: E402
from evaluation.traction_conditions import TRACTION_CONDITION_IDS  # noqa: E402


def write_summary(path, gains):
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=["f_hz", "fft_gain_db", "valid"],
        )
        writer.writeheader()
        for frequency, gain in gains:
            writer.writerow(
                {"f_hz": frequency, "fft_gain_db": gain, "valid": True}
            )


def write_reference(path, rows):
    fields = [
        "f_hz",
        "run_1_gain_db",
        "run_2_gain_db",
        "run_3_gain_db",
        "mean_gain_db",
        "sample_std_gain_db",
        "inverse_variance_weight",
        "normalized_weight",
    ]
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_frf_cost_uses_inverse_variance_weighted_squared_db_error(tmp_path):
    real_reference = tmp_path / "real_reference.csv"
    sim = tmp_path / "sim.csv"
    write_reference(
        real_reference,
        [
            {
                "f_hz": 1.0, "run_1_gain_db": -1.0, "run_2_gain_db": 0.0,
                "run_3_gain_db": 1.0, "mean_gain_db": 0.0,
                "sample_std_gain_db": 1.0,
                "inverse_variance_weight": 1.0,
                "normalized_weight": 16.0 / 17.0,
            },
            {
                "f_hz": 2.0, "run_1_gain_db": -4.0, "run_2_gain_db": 0.0,
                "run_3_gain_db": 4.0, "mean_gain_db": 0.0,
                "sample_std_gain_db": 4.0,
                "inverse_variance_weight": 1.0 / 16.0,
                "normalized_weight": 1.0 / 17.0,
            },
        ],
    )
    write_summary(sim, [(1.0, 2.0), (2.0, 4.0)])

    metric = evaluate_frf_offline(sim, real_reference)

    assert metric.valid is True
    # Variances are 1 and 16 dB^2; normalized weights are 16/17 and 1/17.
    assert metric.J_frf == pytest.approx(
        (16.0 / 17.0) * 2.0**2 + (1.0 / 17.0) * 4.0**2
    )
    assert metric.matched_frequencies == [1.0, 2.0]
    assert metric.cost_method == "inverse_real_sample_variance_weighted_mse_db2_v1"
    assert metric.real_replicate_count == 3
    assert metric.real_reference_path == str(real_reference)
    assert metric.diagnostics["real_reference_sha256"] == hashlib.sha256(
        real_reference.read_bytes()
    ).hexdigest()
    assert metric.gain_db_real == pytest.approx([0.0, 0.0])
    assert metric.gain_db_real_sample_std == pytest.approx([1.0, 4.0])
    assert metric.normalized_frequency_weights == pytest.approx(
        [16.0 / 17.0, 1.0 / 17.0]
    )
    first = metric.per_frequency_error[0]
    assert first["gain_db_real_replicates"] == pytest.approx([-1.0, 0.0, 1.0])
    assert {
        key: value for key, value in first.items()
        if key not in {
            "gain_db_real_replicates",
            "frozen_seven_frequency_weight",
        }
    } == pytest.approx(
        {
            "f_hz": 1.0,
            "gain_db_sim": 2.0,
            "gain_db_real_mean": 0.0,
            "gain_db_real_sample_std": 1.0,
            "error_db": 2.0,
            "squared_error_db2": 4.0,
            "inverse_variance_weight": 1.0,
            "normalized_weight": 16.0 / 17.0,
            "weighted_squared_error_db2": 64.0 / 17.0,
        }
    )


def test_frf_cost_rejects_frozen_normalized_weight_mismatch(tmp_path):
    real_reference = tmp_path / "real_reference.csv"
    sim = tmp_path / "sim.csv"
    write_reference(
        real_reference,
        [{
            "f_hz": 1.0, "run_1_gain_db": -1.0, "run_2_gain_db": 0.0,
            "run_3_gain_db": 1.0, "mean_gain_db": 0.0,
            "sample_std_gain_db": 1.0, "inverse_variance_weight": 1.0,
            "normalized_weight": 0.5,
        }],
    )
    write_summary(sim, [(1.0, 0.0), (2.0, -1.0)])

    metric = evaluate_frf_offline(sim, real_reference)

    assert metric.valid is False
    assert metric.J_frf is None
    assert metric.reject_reason == "real_reference_normalized_weight_mismatch"


def test_frf_cost_rejects_zero_real_sample_std(tmp_path):
    real_reference = tmp_path / "real_reference.csv"
    sim = tmp_path / "sim.csv"
    write_reference(
        real_reference,
        [{
            "f_hz": 1.0, "run_1_gain_db": -1.0, "run_2_gain_db": -1.0,
            "run_3_gain_db": -1.0, "mean_gain_db": -1.0,
            "sample_std_gain_db": 0.0, "inverse_variance_weight": 0.0,
            "normalized_weight": 1.0,
        }],
    )
    write_summary(sim, [(1.0, 0.0)])

    metric = evaluate_frf_offline(sim, real_reference)

    assert metric.valid is False
    assert metric.J_frf is None
    assert metric.reject_reason == "nonpositive_real_gain_sample_std"


def test_frf_cost_rejects_precomputed_mean_mismatch(tmp_path):
    real_reference = tmp_path / "real_reference.csv"
    sim = tmp_path / "sim.csv"
    write_reference(
        real_reference,
        [{
            "f_hz": 1.0, "run_1_gain_db": -1.0, "run_2_gain_db": 0.0,
            "run_3_gain_db": 1.0, "mean_gain_db": 0.1,
            "sample_std_gain_db": 1.0, "inverse_variance_weight": 1.0,
            "normalized_weight": 1.0,
        }],
    )
    write_summary(sim, [(1.0, 0.0)])

    metric = evaluate_frf_offline(sim, real_reference)

    assert metric.valid is False
    assert metric.J_frf is None
    assert metric.reject_reason == "real_reference_mean_gain_mismatch"


def test_managed_dry_run_orders_tendon_x_y_cleanup_then_plain_frf():
    config = load_config().with_overrides(
        {"frf_experiment": {"frequencies": [1.0]}}
    )

    plan = build_managed_dry_run(config, Path("/candidate"))

    expected = []
    for condition_id in TRACTION_CONDITION_IDS:
        expected.extend(
            [f"traction_{condition_id}", f"cleanup_{condition_id}"]
        )
    expected.extend(
        ["frf_1p000000000hz", "cleanup_frf_1p000000000hz"]
    )
    assert [step["phase"] for step in plan["steps"]] == expected
    serialized = str(plan).lower()
    assert "spring" not in serialized
    frf_launch = next(
        step["launch_command"]
        for step in plan["steps"]
        if step["phase"].startswith("frf_")
    )
    assert any("rbkairos.parameterized.xml" in item for item in frf_launch)
    assert "traction_tendon_enabled:=false" in frf_launch
