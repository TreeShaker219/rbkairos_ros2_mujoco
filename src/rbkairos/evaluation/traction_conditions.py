from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class TractionCondition:
    condition_id: str
    axis: str
    speed_mps: float


TRACTION_CONDITIONS = (
    TractionCondition("x_v0p01", "x", 0.01),
    TractionCondition("x_v0p05", "x", 0.05),
    TractionCondition("x_v0p10", "x", 0.10),
    TractionCondition("y_v0p01", "y", 0.01),
    TractionCondition("y_v0p05", "y", 0.05),
    TractionCondition("y_v0p10", "y", 0.10),
)
TRACTION_CONDITION_IDS = tuple(item.condition_id for item in TRACTION_CONDITIONS)
TRACTION_CONDITION_BY_ID = {
    item.condition_id: item for item in TRACTION_CONDITIONS
}


def condition_axis(condition_id: str) -> str:
    try:
        return TRACTION_CONDITION_BY_ID[str(condition_id)].axis
    except KeyError as exc:
        raise ValueError(f"unknown traction condition: {condition_id}") from exc


def condition_speed_mps(condition_id: str) -> float:
    try:
        return TRACTION_CONDITION_BY_ID[str(condition_id)].speed_mps
    except KeyError as exc:
        raise ValueError(f"unknown traction condition: {condition_id}") from exc


def validate_condition_mapping(
    mapping: Mapping[str, Any],
    *,
    label: str,
) -> None:
    actual = tuple(str(key) for key in mapping)
    if set(actual) != set(TRACTION_CONDITION_IDS):
        missing = sorted(set(TRACTION_CONDITION_IDS) - set(actual))
        extra = sorted(set(actual) - set(TRACTION_CONDITION_IDS))
        raise ValueError(
            f"{label} must contain exactly six canonical traction conditions: "
            f"missing={missing}, extra={extra}"
        )


def validate_experiment_contract(
    condition_id: str,
    experiment: Mapping[str, Any],
) -> None:
    axis = condition_axis(condition_id)
    speed = condition_speed_mps(condition_id)
    if str(experiment.get("axis", "")) != axis:
        raise ValueError(f"traction_experiments.{condition_id}.axis mismatch")
    velocity_key = f"velocity_{axis}"
    try:
        actual_speed = float(experiment.get(velocity_key))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"traction_experiments.{condition_id}.{velocity_key} is invalid"
        ) from exc
    if not math.isclose(actual_speed, speed, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"traction_experiments.{condition_id}.{velocity_key} must be {speed}"
        )
    cross_key = "velocity_y" if axis == "x" else "velocity_x"
    if experiment.get(cross_key) not in (None, 0, 0.0):
        raise ValueError(
            f"traction_experiments.{condition_id}.{cross_key} must be absent or zero"
        )
