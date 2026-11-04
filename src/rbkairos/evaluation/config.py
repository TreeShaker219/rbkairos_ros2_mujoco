from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from evaluation.costs import validate_frozen_cost_mapping
from evaluation.traction_conditions import (
    TRACTION_CONDITION_IDS,
    validate_condition_mapping,
    validate_experiment_contract,
)


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "evaluation" / "default.yaml"


def _reject_unknown_keys(
    overlay: dict[str, Any],
    reference: dict[str, Any],
    location: str = "config",
) -> None:
    for key, value in overlay.items():
        key_text = str(key)
        if key not in reference:
            raise ValueError(f"unknown evaluation config key: {location}.{key_text}")
        reference_value = reference[key]
        if isinstance(value, dict) and isinstance(reference_value, dict):
            _reject_unknown_keys(value, reference_value, f"{location}.{key_text}")


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def _as_path(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value))


def _as_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


@dataclass
class EvaluationConfig:
    raw: dict[str, Any]

    def __post_init__(self) -> None:
        self.validate_cost_contract()

    @property
    def mode(self) -> str:
        return str(self.raw.get("mode", "offline"))

    @property
    def output_root(self) -> Path:
        return Path(str(self.raw.get("output_root", "data/evaluation")))

    @property
    def tendon_weight(self) -> float:
        cost = self.raw.get("cost", {})
        return float(cost["tendon_weight"])

    @property
    def frf_weight(self) -> float:
        cost = self.raw.get("cost", {})
        return float(cost["frf_weight"])

    def validate_cost_contract(self) -> None:
        """Reject any drift from the frozen equal-weight outer cost."""
        validate_frozen_cost_mapping(self.raw.get("cost"), label="evaluation")

    @property
    def reuse_policy(self) -> str:
        return str(self.raw.get("reuse_policy", "fresh"))

    @property
    def traction_real_forces(self) -> dict[str, float | None]:
        raw = self.raw.get("references", {}).get("traction_real_forces_n", {})
        if not isinstance(raw, dict):
            raise ValueError("references.traction_real_forces_n must be a mapping")
        validate_condition_mapping(raw, label="references.traction_real_forces_n")
        return {
            condition_id: _as_float(raw.get(condition_id))
            for condition_id in TRACTION_CONDITION_IDS
        }

    @property
    def traction_sim_run_dirs(self) -> dict[str, Path | None]:
        raw = self.raw.get("offline", {}).get("traction_sim_run_dirs", {})
        if not isinstance(raw, dict):
            raise ValueError("offline.traction_sim_run_dirs must be a mapping")
        validate_condition_mapping(raw, label="offline.traction_sim_run_dirs")
        return {
            condition_id: _as_path(raw.get(condition_id))
            for condition_id in TRACTION_CONDITION_IDS
        }

    @property
    def frf_sim_summary(self) -> Path | None:
        return _as_path(self.raw.get("offline", {}).get("frf_sim_summary"))

    @property
    def frf_real_reference(self) -> Path | None:
        return _as_path(
            self.raw.get("references", {}).get("frf_real_reference")
        )

    @property
    def traction_analyzer(self) -> dict[str, Any]:
        return dict(self.raw.get("analyzers", {}).get("traction", {}))

    @property
    def frf_analyzer(self) -> dict[str, Any]:
        return dict(self.raw.get("analyzers", {}).get("frf", {}))

    @property
    def runtime_assets(self) -> dict[str, Any]:
        return dict(self.raw.get("runtime_assets", {}))

    @property
    def managed(self) -> dict[str, Any]:
        return dict(self.raw.get("managed", {}))

    @property
    def traction_experiments(self) -> dict[str, dict[str, Any]]:
        raw_experiments = self.raw.get("traction_experiments")
        if not isinstance(raw_experiments, dict):
            raise ValueError("traction_experiments must be a mapping")
        validate_condition_mapping(raw_experiments, label="traction_experiments")
        result: dict[str, dict[str, Any]] = {}
        for condition_id in TRACTION_CONDITION_IDS:
            value = raw_experiments.get(condition_id)
            if not isinstance(value, dict):
                raise ValueError(
                    f"traction_experiments.{condition_id} must be a mapping"
                )
            validate_experiment_contract(condition_id, value)
            result[condition_id] = dict(value)
        return result

    @property
    def frf_experiment(self) -> dict[str, Any]:
        return dict(self.raw.get("frf_experiment", {}))

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.raw)

    def with_overrides(self, overrides: dict[str, Any]) -> "EvaluationConfig":
        return EvaluationConfig(_deep_merge(self.raw, overrides))


def load_config(path: Path | None = None) -> EvaluationConfig:
    default_data = _read_yaml(DEFAULT_CONFIG)
    if path is None:
        return EvaluationConfig(default_data)
    overlay = _read_yaml(path)
    _reject_unknown_keys(overlay, default_data)
    return EvaluationConfig(_deep_merge(default_data, overlay))
