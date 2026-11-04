from __future__ import annotations

import csv
import json
from pathlib import Path
import struct
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from evaluation.costs import CostBreakdown  # noqa: E402
from evaluation.evaluators import TractionMetric  # noqa: E402
from evaluation.parameters import EvaluationParameters  # noqa: E402
from evaluation.result import EvaluationResult  # noqa: E402
from evaluation.traction_conditions import TRACTION_CONDITIONS  # noqa: E402
from evaluation.traction_outputs import TRACTION_OUTPUT_FILENAMES  # noqa: E402


FORCES = {
    "x_v0p01": 84.47160575487013,
    "x_v0p05": 233.32042027487194,
    "x_v0p10": 419.38007406074706,
    "y_v0p01": 84.47155126015626,
    "y_v0p05": 233.31981159947512,
    "y_v0p10": 419.3720710942022,
}


def _metric(condition_id: str, *, valid: bool = True) -> TractionMetric:
    force = FORCES[condition_id]
    return TractionMetric(
        F_traction_real=force,
        F_traction_sim=force,
        relative_error=0.0,
        J_tendon_point=0.0,
        valid=valid,
        reject_reason="" if valid else "synthetic_invalid",
        run_dir=f"/runs/{condition_id}",
        summary_path=f"/runs/{condition_id}/traction_tendon_summary.json",
        diagnostics={"axis": condition_id[0]},
    )


def _result(output_dir: Path, traction: dict[str, TractionMetric]) -> EvaluationResult:
    return EvaluationResult(
        parameters=EvaluationParameters(kp=30.0, ki=0.5, i_max=3.0),
        cost=CostBreakdown(
            J_tendon_points={item.condition_id: 0.0 for item in TRACTION_CONDITIONS},
            J_tendon_x=0.0,
            J_tendon_y=0.0,
            J_tendon=0.0,
            J_frf=None,
            J_total=None,
            tendon_weight=0.5,
            frf_weight=0.5,
            valid=False,
            reject_reason="missing_frf",
        ),
        traction=traction,
        frf=None,
        provenance={"experiment_scope": "tendon"},
        output_dir=output_dir,
    )


def _png_size(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert payload[12:16] == b"IHDR"
    return struct.unpack(">II", payload[16:24])


def test_result_json_emits_selected_force_csv_and_axis_plots(tmp_path):
    traction = {
        item.condition_id: _metric(item.condition_id)
        for item in TRACTION_CONDITIONS
    }
    result = _result(tmp_path / "candidate", traction)

    result.write_json()

    assert (result.output_dir / "evaluation_result.json").is_file()
    assert all((result.output_dir / name).is_file() for name in TRACTION_OUTPUT_FILENAMES)
    with (result.output_dir / "traction_selected_force.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["condition_id"] for row in rows] == [
        item.condition_id for item in TRACTION_CONDITIONS
    ]
    assert [float(row["selected_force_n"]) for row in rows] == [
        FORCES[item.condition_id] for item in TRACTION_CONDITIONS
    ]
    assert {row["selected_force_source"] for row in rows} == {
        "steady_tow_force_n"
    }
    assert {row["valid"] for row in rows} == {"True"}
    assert _png_size(result.output_dir / "traction_force_x.png") == (1200, 800)
    assert _png_size(result.output_dir / "traction_force_y.png") == (1200, 800)


def test_incomplete_or_invalid_traction_does_not_emit_candidate_outputs(tmp_path):
    complete = {
        item.condition_id: _metric(item.condition_id)
        for item in TRACTION_CONDITIONS
    }
    incomplete = dict(complete)
    incomplete.pop("y_v0p10")
    invalid = dict(complete)
    invalid["y_v0p10"] = _metric("y_v0p10", valid=False)

    for label, traction in (("incomplete", incomplete), ("invalid", invalid)):
        result = _result(tmp_path / label, traction)
        result.write_json()
        payload = json.loads(
            (result.output_dir / "evaluation_result.json").read_text(encoding="utf-8")
        )
        assert payload["traction"] is not None
        assert not any(
            (result.output_dir / name).exists()
            for name in TRACTION_OUTPUT_FILENAMES
        )
