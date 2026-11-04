from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import struct
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from evaluation.costs import CostBreakdown  # noqa: E402
from evaluation.evaluators import FRF_COST_METHOD, FrfMetric  # noqa: E402
from evaluation.frf_outputs import (  # noqa: E402
    FRF_OUTPUT_FILENAMES,
    _gain_tick_labels,
    _unique_gain_ticks,
)
from evaluation.parameters import EvaluationParameters  # noqa: E402
from evaluation.result import EvaluationResult  # noqa: E402
from evaluation.traction_conditions import TRACTION_CONDITION_IDS  # noqa: E402


FREQUENCIES = [0.2, 0.3, 0.5, 1.0, 1.8, 2.8, 4.0]
GAINS_DB = [
    3.255004552,
    3.238972774,
    3.285850382,
    3.072574900,
    0.758866882,
    -3.480205525,
    -8.303183276,
]


def _metric(*, valid: bool = True) -> FrfMetric:
    return FrfMetric(
        J_frf=0.25,
        cost_method=FRF_COST_METHOD,
        real_replicate_count=3,
        real_reference_path="/real/reference.csv",
        matched_frequencies=list(FREQUENCIES),
        gain_db_real=[0.0] * len(FREQUENCIES),
        gain_db_real_sample_std=[1.0] * len(FREQUENCIES),
        gain_db_sim=list(GAINS_DB),
        normalized_frequency_weights=[1.0 / len(FREQUENCIES)] * len(FREQUENCIES),
        per_frequency_error=[
            {"f_hz": frequency, "gain_db_sim": gain}
            for frequency, gain in zip(FREQUENCIES, GAINS_DB)
        ],
        valid_frequency_count=len(FREQUENCIES),
        rejected_frequency_count=0,
        valid=valid,
        reject_reason="" if valid else "synthetic_invalid",
        analysis_dir="/source/frf/analysis",
        summary_path="/source/frf/analysis/frequency_fft_summary.csv",
        diagnostics={},
    )


def _result(output_dir: Path, frf: FrfMetric) -> EvaluationResult:
    return EvaluationResult(
        parameters=EvaluationParameters(kp=30.0, ki=2.0, i_max=20.0),
        cost=CostBreakdown(
            J_tendon_points={condition_id: None for condition_id in TRACTION_CONDITION_IDS},
            J_tendon_x=None,
            J_tendon_y=None,
            J_tendon=None,
            J_frf=frf.J_frf,
            J_total=None,
            tendon_weight=0.5,
            frf_weight=0.5,
            valid=False,
            reject_reason="missing_tendon",
        ),
        traction=None,
        frf=frf,
        provenance={"experiment_scope": "frf"},
        output_dir=output_dir,
        frf_reused_from="kp30_ki2_imax5_source",
        frf_source_raw_data_paths=["/source/raw/f_0p200Hz"],
    )


def _png_size(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert payload[12:16] == b"IHDR"
    return struct.unpack(">II", payload[16:24])


def test_result_json_emits_selected_gain_csv_and_seven_point_plot(tmp_path):
    result = _result(tmp_path / "candidate", _metric())

    result.write_json()

    assert (result.output_dir / "evaluation_result.json").is_file()
    assert all((result.output_dir / name).is_file() for name in FRF_OUTPUT_FILENAMES)
    with (result.output_dir / "frf_selected_gain.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["f_hz"]) for row in rows] == FREQUENCIES
    assert [float(row["fft_gain_db"]) for row in rows] == GAINS_DB
    assert {row["selected_gain_source"] for row in rows} == {
        "FrfMetric.gain_db_sim"
    }
    assert {row["valid"] for row in rows} == {"True"}
    assert {row["frf_reused_from"] for row in rows} == {
        "kp30_ki2_imax5_source"
    }
    assert {row["summary_path"] for row in rows} == {
        "/source/frf/analysis/frequency_fft_summary.csv"
    }
    assert all(
        json.loads(row["source_raw_data_paths_json"])
        == ["/source/raw/f_0p200Hz"]
        for row in rows
    )
    assert _png_size(result.output_dir / "frf_gain.png") == (1200, 800)


def test_gain_axis_ticks_include_zero_and_every_distinct_gain():
    ticks = _unique_gain_ticks(GAINS_DB)
    labels = _gain_tick_labels(ticks)

    assert ticks == sorted([0.0, *GAINS_DB])
    assert len(labels) == len(ticks)
    assert len(set(labels)) == len(labels)
    assert labels[ticks.index(0.0)] == "0"


def test_incomplete_or_invalid_frf_does_not_emit_candidate_outputs(tmp_path):
    invalid = _metric(valid=False)
    incomplete = replace(
        _metric(),
        matched_frequencies=FREQUENCIES[:-1],
        gain_db_sim=GAINS_DB[:-1],
        valid_frequency_count=len(FREQUENCIES) - 1,
    )

    for label, frf in (("invalid", invalid), ("incomplete", incomplete)):
        result = _result(tmp_path / label, frf)
        result.write_json()
        assert not any(
            (result.output_dir / name).exists()
            for name in FRF_OUTPUT_FILENAMES
        )
