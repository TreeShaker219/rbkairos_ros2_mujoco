from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any


FIXED_MU_LOW = 0.01
FIXED_MU_HIGH = 0.7
FIXED_EFFORT_LIMIT_NM = 100.0
FIXED_KD = 0.0


def fixed_parameters_dict() -> dict[str, float | str]:
    return {
        "mu_low": FIXED_MU_LOW,
        "mu_high": FIXED_MU_HIGH,
        "effort_limit_nm": FIXED_EFFORT_LIMIT_NM,
        "kd": FIXED_KD,
        "i_min_rule": "-i_max",
    }


def _finite_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")


def _finite_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")


def format_float_token(value: float, precision: int = 4) -> str:
    text = f"{value:.{precision}g}"
    return text.replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class EvaluationParameters:
    """Candidate optimization variables theta = (Kp, Ki, i_max)."""

    kp: float
    ki: float
    i_max: float

    def validate(self) -> None:
        _finite_positive("kp", float(self.kp))
        _finite_nonnegative("ki", float(self.ki))
        _finite_positive("i_max", float(self.i_max))

    def to_dict(self) -> dict[str, float]:
        return {
            "kp": float(self.kp),
            "ki": float(self.ki),
            "i_max": float(self.i_max),
        }

    def label(self) -> str:
        return (
            f"kp{format_float_token(float(self.kp))}_"
            f"ki{format_float_token(float(self.ki))}_"
            f"imax{format_float_token(float(self.i_max))}"
        )

    def fingerprint(self, extra_payload: dict[str, Any] | None = None) -> str:
        payload = {"parameters": self.to_dict()}
        if extra_payload:
            payload["context"] = extra_payload
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]
