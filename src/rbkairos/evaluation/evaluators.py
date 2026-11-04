from __future__ import annotations

import csv
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.costs import compute_traction_cost


FRF_COST_METHOD = "inverse_real_sample_variance_weighted_mse_db2_v1"
FRF_REAL_REPLICATE_COUNT = 3

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _float_or_nan(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        text = str(value).strip()
        if text == "":
            return math.nan
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def _bool_from_cell(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "nan"}
    return bool(value)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def _sha256_or_none(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frf_gain_db(row: dict[str, Any]) -> tuple[float, str | None]:
    for field in ("fft_gain_db", "gain_db"):
        value = _float_or_nan(row.get(field))
        if math.isfinite(value):
            return value, field
    return math.nan, None


def _valid_frf_summary_rows(rows: list[dict[str, Any]]) -> dict[float, dict[str, Any]]:
    valid: dict[float, dict[str, Any]] = {}
    for row in rows:
        frequency = _float_or_nan(row.get("f_hz"))
        gain_db, gain_field = _frf_gain_db(row)
        if (
            math.isfinite(frequency)
            and math.isfinite(gain_db)
            and _bool_from_cell(row.get("valid"), default=False)
        ):
            normalized = dict(row)
            normalized["gain_db_evaluation"] = gain_db
            normalized["gain_db_source_field"] = gain_field
            valid[round(frequency, 9)] = normalized
    return valid


@dataclass
class TractionMetric:
    F_traction_real: float
    F_traction_sim: float
    relative_error: float | None
    J_tendon_point: float | None
    valid: bool
    reject_reason: str
    run_dir: str
    summary_path: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "F_traction_real": self.F_traction_real,
            "F_traction_sim": self.F_traction_sim,
            "relative_error": self.relative_error,
            "J_tendon_point": self.J_tendon_point,
            "valid": self.valid,
            "reject_reason": self.reject_reason,
            "run_dir": self.run_dir,
            "summary_path": self.summary_path,
            "diagnostics": self.diagnostics,
        }


@dataclass
class FrfMetric:
    J_frf: float | None
    cost_method: str
    real_replicate_count: int
    real_reference_path: str
    matched_frequencies: list[float]
    gain_db_real: list[float]
    gain_db_real_sample_std: list[float]
    gain_db_sim: list[float]
    normalized_frequency_weights: list[float]
    per_frequency_error: list[dict[str, Any]]
    valid_frequency_count: int
    rejected_frequency_count: int
    valid: bool
    reject_reason: str
    analysis_dir: str
    summary_path: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "J_frf": self.J_frf,
            "cost_method": self.cost_method,
            "real_replicate_count": self.real_replicate_count,
            "real_reference_path": self.real_reference_path,
            "matched_frequencies": self.matched_frequencies,
            "gain_db_real": self.gain_db_real,
            "gain_db_real_sample_std": self.gain_db_real_sample_std,
            "gain_db_sim": self.gain_db_sim,
            "normalized_frequency_weights": self.normalized_frequency_weights,
            "per_frequency_error": self.per_frequency_error,
            "valid_frequency_count": self.valid_frequency_count,
            "rejected_frequency_count": self.rejected_frequency_count,
            "valid": self.valid,
            "reject_reason": self.reject_reason,
            "analysis_dir": self.analysis_dir,
            "summary_path": self.summary_path,
            "diagnostics": self.diagnostics,
        }


def evaluate_traction_offline(
    run_dir: Path,
    F_traction_real: float,
    *,
    axis: str,
    steady_window: float = 2.0,
    steady_window_step: float = 0.25,
    min_observation: float = 2.0,
    required_stable_windows: int = 4,
    max_wait: float = 60.0,
    min_tendon_active_ratio: float = 0.95,
    max_force_cv: float = 0.05,
    max_force_slope: float = 0.5,
    max_constraint_extension_m: float = 0.005,
    max_steady_base_velocity_mps: float = 0.005,
    max_i_term_slope: float = 0.005,
    min_i_clamp_ratio: float = 0.95,
    reject_on_wheel_saturation: bool = False,
    wheel_saturation_ratio_limit: float = 0.99,
    strict: bool = False,
    write_summary: bool = True,
    plots: bool = False,
) -> TractionMetric:
    import traction_tendon_analysis

    summary = traction_tendon_analysis.analyze_traction_tendon_run(
        run_dir,
        requested_axis=axis,
        steady_window_s=float(steady_window),
        steady_window_step_s=float(steady_window_step),
        min_observation_s=float(min_observation),
        required_stable_windows=int(required_stable_windows),
        max_wait_s=float(max_wait),
        min_tendon_active_ratio=float(min_tendon_active_ratio),
        max_force_cv=float(max_force_cv),
        max_force_slope_n_s=float(max_force_slope),
        max_constraint_extension_m=float(max_constraint_extension_m),
        max_steady_base_velocity_mps=float(max_steady_base_velocity_mps),
        max_i_term_slope_nm_s=float(max_i_term_slope),
        min_i_clamp_ratio=float(min_i_clamp_ratio),
        write_outputs=bool(write_summary),
        make_plots=bool(plots),
    )

    F_traction_sim = _float_or_nan(summary.get("steady_tow_force_n"))
    wheel_saturation_ratio = _float_or_nan(summary.get("wheel_saturation_ratio"))
    valid = _bool_from_cell(summary.get("valid"), default=False)
    reject_reason = str(summary.get("reject_reason") or "")
    J_tendon_point: float | None = None
    relative_error: float | None = None

    if not valid:
        reject_reason = reject_reason or "traction_run_marked_invalid"
    elif not math.isfinite(F_traction_sim):
        valid = False
        reject_reason = "traction_steady_tow_force_nonfinite"
    elif (
        bool(reject_on_wheel_saturation)
        and math.isfinite(wheel_saturation_ratio)
        and wheel_saturation_ratio >= float(wheel_saturation_ratio_limit)
    ):
        valid = False
        reject_reason = "wheel_qfrc_limit_reached"
    elif not math.isfinite(float(F_traction_real)) or float(F_traction_real) == 0.0:
        valid = False
        reject_reason = "traction_real_force_invalid"
    else:
        relative_error = (F_traction_sim - float(F_traction_real)) / float(F_traction_real)
        J_tendon_point = compute_traction_cost(
            F_traction_sim, float(F_traction_real)
        )

    diagnostics = {
        key: summary.get(key)
        for key in [
            "axis",
            "measurement_method",
            "steady_method",
            "steady_converged",
            "time_to_stable_s",
            "stable_window_count",
            "peak_tow_force_n",
            "steady_tow_force_mean_n",
            "steady_tow_force_std_n",
            "steady_tow_force_cv",
            "steady_axis_force_n",
            "steady_cross_axis_force_n",
            "tendon_active_ratio",
            "time_to_tendon_active_s",
            "max_constraint_extension_m",
            "steady_base_velocity_axis_mps",
            "max_base_displacement_axis_m",
            "max_cross_axis_displacement_m",
            "max_yaw_deg_observed",
            "max_pitch_deg_observed",
            "max_roll_deg_observed",
            "wheel_qfrc_applied_max",
            "wheel_saturation_ratio",
            "limit_mode",
            "warnings",
            "analysis",
        ]
    }
    diagnostics["strict"] = bool(strict)
    diagnostics["reject_on_wheel_saturation"] = bool(reject_on_wheel_saturation)
    diagnostics["wheel_saturation_ratio_limit"] = float(wheel_saturation_ratio_limit)

    return TractionMetric(
        F_traction_real=float(F_traction_real),
        F_traction_sim=F_traction_sim,
        relative_error=relative_error,
        J_tendon_point=J_tendon_point,
        valid=bool(valid),
        reject_reason=reject_reason,
        run_dir=str(run_dir),
        summary_path=str(run_dir / "traction_tendon_summary.json"),
        diagnostics=diagnostics,
    )


def _frf_reference_rows(
    rows: list[dict[str, Any]],
) -> tuple[dict[float, dict[str, Any]], str]:
    if not rows:
        return {}, "real_reference_empty"
    replicate_fields = sorted(
        field
        for field in rows[0]
        if field.startswith("run_") and field.endswith("_gain_db")
    )
    if len(replicate_fields) != FRF_REAL_REPLICATE_COUNT:
        return {}, "real_reference_replicate_count_mismatch"

    reference: dict[float, dict[str, Any]] = {}
    for row in rows:
        frequency = _float_or_nan(row.get("f_hz"))
        mean_gain = _float_or_nan(row.get("mean_gain_db"))
        sample_std = _float_or_nan(row.get("sample_std_gain_db"))
        inverse_weight = _float_or_nan(row.get("inverse_variance_weight"))
        frozen_normalized_weight = _float_or_nan(row.get("normalized_weight"))
        replicate_gains = [
            _float_or_nan(row.get(field)) for field in replicate_fields
        ]
        if not math.isfinite(frequency) or frequency <= 0.0:
            return {}, "real_reference_frequency_invalid"
        frequency = round(frequency, 9)
        if frequency in reference:
            return {}, "real_reference_frequency_duplicate"
        if not math.isfinite(mean_gain) or not all(
            math.isfinite(value) for value in replicate_gains
        ):
            return {}, "real_reference_gain_nonfinite"
        computed_mean = sum(replicate_gains) / FRF_REAL_REPLICATE_COUNT
        if not math.isclose(
            mean_gain,
            computed_mean,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            return {}, "real_reference_mean_gain_mismatch"
        if not math.isfinite(sample_std) or sample_std <= 0.0:
            return {}, "nonpositive_real_gain_sample_std"
        squared_deviation_sum = sum(
            (value - computed_mean) ** 2 for value in replicate_gains
        )
        computed_sample_std = math.sqrt(
            squared_deviation_sum / (FRF_REAL_REPLICATE_COUNT - 1)
        )
        if not math.isclose(
            sample_std,
            computed_sample_std,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            return {}, "real_reference_sample_std_mismatch"
        if not math.isfinite(inverse_weight) or inverse_weight <= 0.0:
            return {}, "real_reference_inverse_variance_weight_invalid"
        if not math.isclose(
            inverse_weight,
            1.0 / (sample_std * sample_std),
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            return {}, "real_reference_inverse_variance_weight_mismatch"
        normalized_weight_is_valid = (
            math.isfinite(frozen_normalized_weight)
            and frozen_normalized_weight > 0.0
        )
        if not normalized_weight_is_valid:
            return {}, "real_reference_normalized_weight_invalid"
        reference[frequency] = {
            "mean_gain_db": mean_gain,
            "sample_std_gain_db": sample_std,
            "inverse_variance_weight": inverse_weight,
            "frozen_normalized_weight": frozen_normalized_weight,
            "replicate_gains_db": replicate_gains,
            "replicate_fields": replicate_fields,
        }

    inverse_weight_sum = sum(
        row["inverse_variance_weight"] for row in reference.values()
    )
    for row in reference.values():
        expected = row["inverse_variance_weight"] / inverse_weight_sum
        if not math.isclose(
            row["frozen_normalized_weight"],
            expected,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            return {}, "real_reference_normalized_weight_mismatch"
    return reference, ""


def _invalid_frf_metric(
    sim_summary_csv: Path,
    real_reference_csv: Path,
    reject_reason: str,
    diagnostics: dict[str, Any],
    *,
    rejected_frequency_count: int = 0,
) -> FrfMetric:
    return FrfMetric(
        J_frf=None,
        cost_method=FRF_COST_METHOD,
        real_replicate_count=FRF_REAL_REPLICATE_COUNT,
        real_reference_path=str(real_reference_csv),
        matched_frequencies=[],
        gain_db_real=[],
        gain_db_real_sample_std=[],
        gain_db_sim=[],
        normalized_frequency_weights=[],
        per_frequency_error=[],
        valid_frequency_count=0,
        rejected_frequency_count=rejected_frequency_count,
        valid=False,
        reject_reason=reject_reason,
        analysis_dir=str(sim_summary_csv.parent),
        summary_path=str(sim_summary_csv),
        diagnostics=diagnostics,
    )


def evaluate_frf_offline(
    sim_summary_csv: Path,
    real_reference_csv: Path,
) -> FrfMetric:
    sim_by_f = _valid_frf_summary_rows(_read_csv_rows(sim_summary_csv))
    real_reference, reference_error = _frf_reference_rows(
        _read_csv_rows(real_reference_csv)
    )
    if reference_error:
        return _invalid_frf_metric(
            sim_summary_csv,
            real_reference_csv,
            reference_error,
            {"sim_valid_frequencies": sorted(sim_by_f)},
        )

    real_frequencies = set(real_reference)
    common = sorted(set(sim_by_f) & real_frequencies)
    rejected_frequency_count = len(set(sim_by_f) ^ real_frequencies)
    if not common:
        return _invalid_frf_metric(
            sim_summary_csv,
            real_reference_csv,
            "no_common_valid_frequencies",
            {
                "sim_valid_frequencies": sorted(sim_by_f),
                "real_valid_frequencies": sorted(real_frequencies),
            },
            rejected_frequency_count=rejected_frequency_count,
        )

    matched_weight_sum = sum(
        real_reference[frequency]["inverse_variance_weight"]
        for frequency in common
    )
    per_frequency_error: list[dict[str, Any]] = []
    gain_db_real: list[float] = []
    gain_db_real_sample_std: list[float] = []
    gain_db_sim: list[float] = []
    normalized_frequency_weights: list[float] = []

    for frequency in common:
        sim_gain = _float_or_nan(sim_by_f[frequency].get("gain_db_evaluation"))
        reference_row = real_reference[frequency]
        real_gain_mean = reference_row["mean_gain_db"]
        real_gain_sample_std = reference_row["sample_std_gain_db"]
        error = sim_gain - real_gain_mean
        squared_error = error * error
        inverse_weight = reference_row["inverse_variance_weight"]
        normalized_weight = inverse_weight / matched_weight_sum
        per_frequency_error.append(
            {
                "f_hz": frequency,
                "gain_db_sim": sim_gain,
                "gain_db_real_replicates": reference_row["replicate_gains_db"],
                "gain_db_real_mean": real_gain_mean,
                "gain_db_real_sample_std": real_gain_sample_std,
                "error_db": error,
                "squared_error_db2": squared_error,
                "inverse_variance_weight": inverse_weight,
                "normalized_weight": normalized_weight,
                "frozen_seven_frequency_weight": reference_row[
                    "frozen_normalized_weight"
                ],
                "weighted_squared_error_db2": (
                    normalized_weight * squared_error
                ),
            }
        )
        gain_db_sim.append(sim_gain)
        gain_db_real.append(real_gain_mean)
        gain_db_real_sample_std.append(real_gain_sample_std)
        normalized_frequency_weights.append(normalized_weight)

    J_frf = sum(
        row["weighted_squared_error_db2"] for row in per_frequency_error
    )
    diagnostics = {
        "cost_method": FRF_COST_METHOD,
        "cost_formula": (
            "sum((1/sample_variance_db2)*error_db^2)"
            "/sum(1/sample_variance_db2)"
        ),
        "real_replicate_count": FRF_REAL_REPLICATE_COUNT,
        "real_reference_path": str(real_reference_csv),
        "real_reference_sha256": _sha256_or_none(real_reference_csv),
        "reference_values_precomputed": True,
        "sim_valid_frequencies": sorted(sim_by_f),
        "real_valid_frequencies": sorted(real_frequencies),
        "gain_db_source_field_sim": [
            sim_by_f[f].get("gain_db_source_field") for f in common
        ],
        "phase_rad_optional_sim": [
            _float_or_nan(sim_by_f[f].get("phase_rad_optional"))
            for f in common
        ],
        "fft_phase_deg_sim": [
            _float_or_nan(sim_by_f[f].get("fft_phase_deg")) for f in common
        ],
        "cmd_fit_r2_sim": [
            _float_or_nan(sim_by_f[f].get("cmd_fit_r2")) for f in common
        ],
        "out_fit_r2_sim": [
            _float_or_nan(sim_by_f[f].get("out_fit_r2")) for f in common
        ],
        "fft_output_fundamental_energy_fraction_sim": [
            _float_or_nan(
                sim_by_f[f].get("fft_output_fundamental_energy_fraction")
            )
            for f in common
        ],
        "thd_2_to_max_sim": [
            _float_or_nan(sim_by_f[f].get("thd_2_to_max")) for f in common
        ],
        "saturation_ratio_sim": [
            _float_or_nan(sim_by_f[f].get("saturation_ratio")) for f in common
        ],
        "contact_force_rms_sim": [
            _float_or_nan(sim_by_f[f].get("contact_force_rms")) for f in common
        ],
    }

    return FrfMetric(
        J_frf=J_frf,
        cost_method=FRF_COST_METHOD,
        real_replicate_count=FRF_REAL_REPLICATE_COUNT,
        real_reference_path=str(real_reference_csv),
        matched_frequencies=common,
        gain_db_real=gain_db_real,
        gain_db_real_sample_std=gain_db_real_sample_std,
        gain_db_sim=gain_db_sim,
        normalized_frequency_weights=normalized_frequency_weights,
        per_frequency_error=per_frequency_error,
        valid_frequency_count=len(common),
        rejected_frequency_count=rejected_frequency_count,
        valid=True,
        reject_reason="",
        analysis_dir=str(sim_summary_csv.parent),
        summary_path=str(sim_summary_csv),
        diagnostics=diagnostics,
    )
