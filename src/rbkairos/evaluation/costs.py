from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from evaluation.traction_conditions import (
    TRACTION_CONDITION_IDS,
    condition_axis,
)


FROZEN_TENDON_WEIGHT = 0.5
FROZEN_FRF_WEIGHT = 0.5
COST_CONTRACT_SCHEMA = "rbkairos_evaluation_cost_contract_v2"


def frozen_cost_contract() -> dict[str, float | str]:
    """Return the immutable outer-cost contract used by formal evaluation."""
    return {
        "schema": COST_CONTRACT_SCHEMA,
        "tendon_weight": FROZEN_TENDON_WEIGHT,
        "frf_weight": FROZEN_FRF_WEIGHT,
        "formula": "J_total=0.5*J_tendon+0.5*J_frf",
    }


def validate_frozen_cost_mapping(
    value: Any,
    *,
    label: str,
) -> dict[str, float]:
    """Validate and normalize the frozen equal-weight cost mapping."""
    expected = {
        "tendon_weight": FROZEN_TENDON_WEIGHT,
        "frf_weight": FROZEN_FRF_WEIGHT,
    }
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(
            f"{label} frozen cost contract mismatch: "
            f"expected exact keys {sorted(expected)}"
        )
    normalized: dict[str, float] = {}
    for name, expected_value in expected.items():
        raw = value[name]
        if isinstance(raw, bool):
            raise ValueError(
                f"{label} frozen cost contract mismatch: {name} must equal "
                f"{expected_value}, not boolean"
            )
        try:
            actual = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} frozen cost contract mismatch: "
                f"{name} must equal {expected_value}"
            ) from exc
        if not math.isfinite(actual) or actual != expected_value:
            raise ValueError(
                f"{label} frozen cost contract mismatch: {name} must equal "
                f"{expected_value}, got {actual}"
            )
        normalized[name] = actual
    return normalized


@dataclass
class CostBreakdown:
    J_tendon_points: dict[str, float | None]
    J_tendon_x: float | None
    J_tendon_y: float | None
    J_tendon: float | None
    J_frf: float | None
    J_total: float | None
    tendon_weight: float
    frf_weight: float
    valid: bool
    reject_reason: str

    def to_dict(self) -> dict[str, float | bool | str | None]:
        return {
            "J_tendon_points": dict(self.J_tendon_points),
            "J_tendon_x": self.J_tendon_x,
            "J_tendon_y": self.J_tendon_y,
            "J_tendon": self.J_tendon,
            "J_frf": self.J_frf,
            "J_total": self.J_total,
            "tendon_weight": self.tendon_weight,
            "frf_weight": self.frf_weight,
            "valid": self.valid,
            "reject_reason": self.reject_reason,
        }


def compute_traction_cost(F_traction_sim: float, F_traction_real: float) -> float:
    if not math.isfinite(F_traction_sim):
        raise ValueError("F_traction_sim must be finite")
    if not math.isfinite(F_traction_real) or F_traction_real == 0.0:
        raise ValueError("F_traction_real must be finite and non-zero")
    return ((F_traction_sim - F_traction_real) / F_traction_real) ** 2


def compute_multispeed_total_cost(
    J_tendon_points: dict[str, float | None],
    J_frf: float | None,
    reject_reasons: list[str] | None = None,
) -> CostBreakdown:
    reasons = [reason for reason in (reject_reasons or []) if reason]
    canonical_points = {
        condition_id: J_tendon_points.get(condition_id)
        for condition_id in TRACTION_CONDITION_IDS
    }
    traction_values_valid = all(
        value is not None and math.isfinite(float(value))
        for value in canonical_points.values()
    )
    frf_valid = J_frf is not None and math.isfinite(float(J_frf))
    axis_values = {
        axis: [
            float(value)
            for condition_id, value in canonical_points.items()
            if condition_axis(condition_id) == axis and value is not None
        ]
        for axis in ("x", "y")
    }
    J_tendon_x = (
        sum(axis_values["x"]) / 3.0 if traction_values_valid else None
    )
    J_tendon_y = (
        sum(axis_values["y"]) / 3.0 if traction_values_valid else None
    )
    J_tendon = (
        sum(float(value) for value in canonical_points.values()) / 6.0
        if traction_values_valid
        else None
    )
    valid = (
        not reasons
        and traction_values_valid
        and frf_valid
        and J_tendon is not None
    )
    if not valid:
        return CostBreakdown(
            J_tendon_points=canonical_points,
            J_tendon_x=J_tendon_x,
            J_tendon_y=J_tendon_y,
            J_tendon=J_tendon,
            J_frf=J_frf,
            J_total=None,
            tendon_weight=FROZEN_TENDON_WEIGHT,
            frf_weight=FROZEN_FRF_WEIGHT,
            valid=False,
            reject_reason="; ".join(reasons) or "invalid_metric",
        )

    return CostBreakdown(
        J_tendon_points={key: float(value) for key, value in canonical_points.items()},
        J_tendon_x=float(J_tendon_x),
        J_tendon_y=float(J_tendon_y),
        J_tendon=J_tendon,
        J_frf=float(J_frf),
        J_total=(
            FROZEN_TENDON_WEIGHT * J_tendon
            + FROZEN_FRF_WEIGHT * float(J_frf)
        ),
        tendon_weight=FROZEN_TENDON_WEIGHT,
        frf_weight=FROZEN_FRF_WEIGHT,
        valid=True,
        reject_reason="",
    )
