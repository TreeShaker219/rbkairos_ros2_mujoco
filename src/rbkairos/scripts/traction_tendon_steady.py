#!/usr/bin/env python3
"""
Shared adaptive steady-state contract for Tendon traction measurements.

The functions in this module are ROS-independent.  The runtime recorder and
offline analyzers intentionally use the same window evaluator so a run cannot
be accepted online and interpreted with different criteria later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
from typing import Any, Mapping, Sequence


STEADY_METHOD = "adaptive_sustained_windows_v1"


class SteadyStateError(ValueError):
    """Raised when adaptive steady-state inputs violate the data contract."""


@dataclass(frozen=True)
class SteadyStateConfig:
    window_s: float = 2.0
    window_step_s: float = 0.25
    min_observation_s: float = 2.0
    required_stable_windows: int = 4
    max_wait_s: float = 60.0
    min_tendon_active_ratio: float = 0.95
    max_force_cv: float = 0.05
    max_abs_force_slope_n_s: float = 0.5
    max_constraint_extension_m: float = 0.005
    max_abs_base_velocity_mps: float = 0.005
    max_abs_i_term_slope_nm_s: float = 0.005
    min_i_clamp_ratio: float = 0.95

    def __post_init__(self) -> None:
        positive = {
            "window_s": self.window_s,
            "window_step_s": self.window_step_s,
            "min_observation_s": self.min_observation_s,
            "max_wait_s": self.max_wait_s,
            "max_force_cv": self.max_force_cv,
            "max_abs_force_slope_n_s": self.max_abs_force_slope_n_s,
            "max_constraint_extension_m": self.max_constraint_extension_m,
            "max_abs_base_velocity_mps": self.max_abs_base_velocity_mps,
            "max_abs_i_term_slope_nm_s": self.max_abs_i_term_slope_nm_s,
        }
        for label, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise SteadyStateError(f"{label} must be finite and positive")
        if self.min_observation_s < self.window_s:
            raise SteadyStateError("min_observation_s must be at least window_s")
        if not isinstance(self.required_stable_windows, int) or self.required_stable_windows < 1:
            raise SteadyStateError("required_stable_windows must be a positive integer")
        minimum_detection = (
            self.min_observation_s
            + (self.required_stable_windows - 1) * self.window_step_s
        )
        if self.max_wait_s + 1e-12 < minimum_detection:
            raise SteadyStateError(
                "max_wait_s is shorter than the earliest persistent-stability decision"
            )
        for label, value in {
            "min_tendon_active_ratio": self.min_tendon_active_ratio,
            "min_i_clamp_ratio": self.min_i_clamp_ratio,
        }.items():
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise SteadyStateError(f"{label} must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SteadyStateError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise SteadyStateError(f"{label} is non-finite")
    return result


def _flag(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and float(value) in (0.0, 1.0):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "1.0", "true"}:
            return True
        if normalized in {"0", "0.0", "false"}:
            return False
    raise SteadyStateError(f"{label} must be a boolean flag")


def linear_slope(times: Sequence[float], values: Sequence[float]) -> float:
    if len(times) != len(values) or len(times) < 2:
        raise SteadyStateError("linear slope needs at least two aligned samples")
    mean_time = statistics.fmean(times)
    mean_value = statistics.fmean(values)
    denominator = sum((value - mean_time) ** 2 for value in times)
    if denominator <= 0.0:
        raise SteadyStateError("linear slope timestamps have zero span")
    return sum(
        (time - mean_time) * (value - mean_value)
        for time, value in zip(times, values)
    ) / denominator


def _window(
    rows: Sequence[Mapping[str, Any]], start_s: float, end_s: float
) -> list[Mapping[str, Any]]:
    tolerance = 1e-9
    return [
        row for row in rows
        if start_s - tolerance
        <= _finite(row.get("t_experiment"), "sample time")
        <= end_s + tolerance
    ]


def _validate_monotonic(rows: Sequence[Mapping[str, Any]], label: str) -> None:
    previous = -math.inf
    for row in rows:
        timestamp = _finite(row.get("t_experiment"), f"{label} time")
        if timestamp + 1e-12 < previous:
            raise SteadyStateError(f"{label} time is not monotonic")
        previous = timestamp


def evaluate_steady_window(
    state_rows: Sequence[Mapping[str, Any]],
    odom_rows: Sequence[Mapping[str, Any]],
    pid_rows: Sequence[Mapping[str, Any]],
    *,
    axis: str,
    start_s: float,
    end_s: float,
    config: SteadyStateConfig,
    wheel_names: Sequence[str],
) -> dict[str, Any]:
    """Evaluate every validity signal over one aligned sliding window."""
    if axis not in ("x", "y"):
        raise SteadyStateError("axis must be x or y")
    if end_s <= start_s:
        raise SteadyStateError("steady window has an invalid interval")
    wheels = tuple(wheel_names)
    if len(wheels) != 4 or len(set(wheels)) != 4:
        raise SteadyStateError("wheel_names must contain exactly four unique wheels")

    state = _window(state_rows, start_s, end_s)
    odom = _window(odom_rows, start_s, end_s)
    pid = _window(pid_rows, start_s, end_s)
    if len(state) < 3 or len(odom) < 3:
        raise SteadyStateError(
            "empty or insufficient steady window in state or odom samples"
        )
    names = {str(row.get("joint_name", "")) for row in pid}
    if names != set(wheels):
        raise SteadyStateError("steady window PID rows must contain exactly four wheels")

    active = [
        row for row in state
        if _flag(row.get("limit_active_flag"), "Tendon active flag")
    ]
    tensions = [_finite(row.get("tension_n"), "Tendon tension") for row in active]
    active_ratio = len(active) / len(state)
    force_mean = statistics.fmean(tensions) if tensions else math.nan
    force_std = statistics.stdev(tensions) if len(tensions) > 1 else 0.0
    force_cv = (
        force_std / abs(force_mean)
        if tensions and not math.isclose(force_mean, 0.0, abs_tol=1e-15)
        else math.inf
    )
    force_slope = (
        linear_slope(
            [_finite(row.get("t_experiment"), "Tendon state time") for row in active],
            tensions,
        )
        if len(active) >= 2 else math.inf
    )
    max_extension = max(
        _finite(row.get("extension_over_upper_limit_m"), "constraint extension")
        for row in state
    )
    velocity_field = "vx" if axis == "x" else "vy"
    mean_base_velocity = statistics.fmean(
        _finite(row.get(velocity_field), f"base {velocity_field}") for row in odom
    )

    per_wheel: dict[str, Any] = {}
    for wheel in wheels:
        rows = [row for row in pid if str(row.get("joint_name", "")) == wheel]
        if len(rows) < 2:
            raise SteadyStateError(f"steady window has insufficient PID samples for {wheel}")
        times = [_finite(row.get("t_experiment"), "PID time") for row in rows]
        i_terms = [_finite(row.get("i_term"), "PID I term") for row in rows]
        i_slope = linear_slope(times, i_terms)
        i_clamp_ratio = sum(
            _flag(row.get("i_clamp_active"), "PID I clamp flag") for row in rows
        ) / len(rows)
        effort_clamp_ratio = sum(
            _flag(row.get("effort_clamp_active"), "PID effort clamp flag") for row in rows
        ) / len(rows)
        per_wheel[wheel] = {
            "sample_count": len(rows),
            "i_slope_nm_s": i_slope,
            "i_clamp_ratio": i_clamp_ratio,
            "effort_clamp_ratio": effort_clamp_ratio,
            "i_stable": (
                abs(i_slope) <= config.max_abs_i_term_slope_nm_s
                or i_clamp_ratio >= config.min_i_clamp_ratio
            ),
        }

    reasons: list[str] = []
    if active_ratio < config.min_tendon_active_ratio:
        reasons.append("tendon_active_ratio")
    if not math.isfinite(force_cv) or force_cv > config.max_force_cv:
        reasons.append("force_cv")
    if abs(force_slope) > config.max_abs_force_slope_n_s:
        reasons.append("force_slope")
    if max_extension > config.max_constraint_extension_m:
        reasons.append("constraint_extension")
    if abs(mean_base_velocity) > config.max_abs_base_velocity_mps:
        reasons.append("base_velocity")
    if not all(item["i_stable"] for item in per_wheel.values()):
        reasons.append("i_term_slope")
    if any(item["effort_clamp_ratio"] > 0.0 for item in per_wheel.values()):
        reasons.append("effort_clamp")

    return {
        "start_s": start_s,
        "end_s": end_s,
        "sample_count": len(state),
        "active_ratio": active_ratio,
        "force_median_n": statistics.median(tensions) if tensions else None,
        "force_mean_n": force_mean,
        "force_std_n": force_std,
        "force_cv": force_cv,
        "force_slope_n_s": force_slope,
        "max_constraint_extension_m": max_extension,
        "mean_base_velocity_axis_mps": mean_base_velocity,
        "per_wheel": per_wheel,
        "stable": not reasons,
        "reasons": reasons,
    }


def _first_sustained_flag(
    rows: Sequence[Mapping[str, Any]], field: str, minimum_duration_s: float = 0.5
) -> float | None:
    start: float | None = None
    previous = -math.inf
    for row in rows:
        timestamp = _finite(row.get("t_experiment"), "PID time")
        if timestamp + 1e-12 < previous:
            raise SteadyStateError("PID time is not monotonic")
        if _flag(row.get(field), field):
            if start is None:
                start = timestamp
            if timestamp - start >= minimum_duration_s - 1e-9:
                return start
        else:
            start = None
        previous = timestamp
    return None


def _summarize_plateau(
    state_rows: Sequence[Mapping[str, Any]],
    pid_rows: Sequence[Mapping[str, Any]],
    *,
    start_s: float,
    end_s: float,
    wheel_names: Sequence[str],
    min_i_clamp_ratio: float,
) -> tuple[float | None, dict[str, Any], bool]:
    plateau_state = _window(state_rows, start_s, end_s)
    active = [
        row for row in plateau_state
        if _flag(row.get("limit_active_flag"), "Tendon active flag")
    ]
    tension = (
        statistics.median(
            _finite(row.get("tension_n"), "Tendon tension") for row in active
        )
        if active else None
    )
    per_wheel: dict[str, Any] = {}
    all_i_clamped = True
    for wheel in wheel_names:
        all_rows = [
            row for row in pid_rows if str(row.get("joint_name", "")) == wheel
        ]
        rows = _window(all_rows, start_s, end_s)
        clamp_start = _first_sustained_flag(all_rows, "i_clamp_active")
        ratio = (
            sum(_flag(row.get("i_clamp_active"), "PID I clamp flag") for row in rows)
            / len(rows)
            if rows else 0.0
        )
        all_i_clamped = all_i_clamped and ratio >= min_i_clamp_ratio
        times = [_finite(row.get("t_experiment"), "PID time") for row in rows]
        terms = [_finite(row.get("i_term"), "PID I term") for row in rows]
        per_wheel[wheel] = {
            "time_to_i_clamp_s": clamp_start,
            "i_clamp_start_experiment_s": clamp_start,
            "plateau_i_term_nm": statistics.median(terms) if terms else None,
            "plateau_i_slope_nm_s": (
                linear_slope(times, terms) if len(times) >= 2 else None
            ),
            "plateau_i_clamp_ratio": ratio,
            "effort_clamp_ratio": (
                sum(
                    _flag(row.get("effort_clamp_active"), "PID effort clamp flag")
                    for row in rows
                ) / len(rows)
                if rows else 0.0
            ),
            "plateau_qfrc_applied_nm": (
                statistics.median(
                    _finite(row.get("qfrc_applied"), "qfrc_applied") for row in rows
                ) if rows else None
            ),
        }
    return tension, per_wheel, all_i_clamped


def analyze_sustained_steady_state(
    state_rows: Sequence[Mapping[str, Any]],
    odom_rows: Sequence[Mapping[str, Any]],
    pid_rows: Sequence[Mapping[str, Any]],
    *,
    axis: str,
    constant_start_s: float,
    constant_end_s: float,
    config: SteadyStateConfig,
    wheel_names: Sequence[str],
) -> dict[str, Any]:
    """Retrospectively require a persistent stable run at the interval tail."""
    if axis not in ("x", "y"):
        raise SteadyStateError("axis must be x or y")
    if constant_end_s <= constant_start_s:
        raise SteadyStateError("constant interval is invalid")
    for rows, label in (
        (state_rows, "Tendon state"),
        (odom_rows, "odom"),
        (pid_rows, "PID"),
    ):
        _validate_monotonic(rows, label)
    constant_state = _window(state_rows, constant_start_s, constant_end_s)
    constant_odom = _window(odom_rows, constant_start_s, constant_end_s)
    constant_pid = _window(pid_rows, constant_start_s, constant_end_s)
    if len(constant_state) < 3 or len(constant_odom) < 3 or len(constant_pid) < 8:
        raise SteadyStateError("constant interval has insufficient samples")
    if {str(row.get("joint_name", "")) for row in constant_pid} != set(wheel_names):
        raise SteadyStateError("constant PID rows must contain exactly four wheels")

    effort_clamp_any = any(
        _flag(row.get("effort_clamp_active"), "PID effort clamp flag")
        for row in constant_pid
    )
    windows: list[dict[str, Any]] = []
    end_s = constant_start_s + config.min_observation_s
    while end_s <= constant_end_s + 1e-9:
        start_s = end_s - config.window_s
        windows.append(evaluate_steady_window(
            constant_state,
            constant_odom,
            constant_pid,
            axis=axis,
            start_s=start_s,
            end_s=end_s,
            config=config,
            wheel_names=wheel_names,
        ))
        end_s += config.window_step_s

    trailing_stable = 0
    for window in reversed(windows):
        if not window["stable"]:
            break
        trailing_stable += 1
    converged = trailing_stable >= config.required_stable_windows and not effort_clamp_any
    stable_index = len(windows) - trailing_stable if converged else None
    stable_start = windows[stable_index]["start_s"] if stable_index is not None else None
    detection_index = (
        stable_index + config.required_stable_windows - 1
        if stable_index is not None else None
    )
    detection_time = (
        windows[detection_index]["end_s"] if detection_index is not None else None
    )
    plateau_force: float | None = None
    per_wheel: dict[str, Any] = {}
    all_wheels_i_clamped = False
    if converged and stable_start is not None:
        plateau_force, per_wheel, all_wheels_i_clamped = _summarize_plateau(
            constant_state,
            constant_pid,
            start_s=stable_start,
            end_s=constant_end_s,
            wheel_names=wheel_names,
            min_i_clamp_ratio=config.min_i_clamp_ratio,
        )

    if effort_clamp_any:
        status = "effort_clamp"
        reject_reasons = ["effort_clamp"]
    elif not converged:
        status = "not_converged"
        reject_reasons = ["not_converged"]
    else:
        status = "converged"
        reject_reasons = []
    return {
        "steady_method": STEADY_METHOD,
        "axis": axis,
        "constant_start_s": constant_start_s,
        "constant_end_s": constant_end_s,
        "status": status,
        "converged": converged,
        "valid": not reject_reasons,
        "reject_reasons": reject_reasons,
        "stable_start_s": stable_start,
        "time_to_stable_s": (
            stable_start - constant_start_s if stable_start is not None else None
        ),
        "convergence_detected_s": detection_time,
        "stable_window_count": trailing_stable,
        "steady_tow_force_n": plateau_force,
        "plateau_force_n": plateau_force,
        "all_wheels_i_clamped": all_wheels_i_clamped,
        "effort_clamp_any": effort_clamp_any,
        "max_constraint_extension_m": max(
            _finite(row.get("extension_over_upper_limit_m"), "constraint extension")
            for row in constant_state
        ),
        "per_wheel": per_wheel,
        "windows": windows,
        "last_window": windows[-1] if windows else None,
        "steady_state_config": config.to_dict(),
    }


class AdaptiveSteadyMonitor:
    """Incremental online monitor with a resettable consecutive-window streak."""

    def __init__(
        self,
        *,
        axis: str,
        constant_start_s: float,
        config: SteadyStateConfig,
        wheel_names: Sequence[str],
    ) -> None:
        if axis not in ("x", "y"):
            raise SteadyStateError("axis must be x or y")
        self.axis = axis
        self.constant_start_s = float(constant_start_s)
        self.config = config
        self.wheel_names = tuple(wheel_names)
        self.state_history: list[Mapping[str, Any]] = []
        self.odom_history: list[Mapping[str, Any]] = []
        self.pid_history: list[Mapping[str, Any]] = []
        self.windows: list[dict[str, Any]] = []
        self.stable_streak_count = 0
        self.reset_count = 0
        self._streak_start_s: float | None = None
        self._next_end_s = self.constant_start_s + config.min_observation_s
        self._status = "collecting"
        self._convergence_detected_s: float | None = None
        self._effort_clamp_any = False

    def add_state(self, row: Mapping[str, Any]) -> None:
        self.state_history.append(row)

    def add_odom(self, row: Mapping[str, Any]) -> None:
        self.odom_history.append(row)

    def add_pid(self, row: Mapping[str, Any]) -> None:
        self.pid_history.append(row)
        if _flag(row.get("effort_clamp_active"), "PID effort clamp flag"):
            self._effort_clamp_any = True

    def advance(self, current_time_s: float) -> dict[str, Any]:
        now = _finite(current_time_s, "monitor time")
        if self._status != "collecting":
            return self.result()
        if self._effort_clamp_any:
            self._status = "effort_clamp"
            return self.result()

        while self._next_end_s <= now + 1e-9 and self._status == "collecting":
            window = evaluate_steady_window(
                self.state_history,
                self.odom_history,
                self.pid_history,
                axis=self.axis,
                start_s=self._next_end_s - self.config.window_s,
                end_s=self._next_end_s,
                config=self.config,
                wheel_names=self.wheel_names,
            )
            self.windows.append(window)
            if window["stable"]:
                if self.stable_streak_count == 0:
                    self._streak_start_s = window["start_s"]
                self.stable_streak_count += 1
                if self.stable_streak_count >= self.config.required_stable_windows:
                    self._status = "converged"
                    self._convergence_detected_s = window["end_s"]
            else:
                if self.stable_streak_count:
                    self.reset_count += 1
                self.stable_streak_count = 0
                self._streak_start_s = None
            self._next_end_s += self.config.window_step_s

        if (
            self._status == "collecting"
            and now - self.constant_start_s >= self.config.max_wait_s - 1e-9
        ):
            self._status = "not_converged"
        return self.result()

    def result(self) -> dict[str, Any]:
        converged = self._status == "converged"
        tension: float | None = None
        per_wheel: dict[str, Any] = {}
        all_i_clamped = False
        if converged and self._streak_start_s is not None:
            tension, per_wheel, all_i_clamped = _summarize_plateau(
                self.state_history,
                self.pid_history,
                start_s=self._streak_start_s,
                end_s=float(self._convergence_detected_s),
                wheel_names=self.wheel_names,
                min_i_clamp_ratio=self.config.min_i_clamp_ratio,
            )
        reject_reasons = (
            [] if converged or self._status == "collecting" else [self._status]
        )
        return {
            "steady_method": STEADY_METHOD,
            "status": self._status,
            "converged": converged,
            "valid": converged,
            "reject_reasons": reject_reasons,
            "stable_start_s": self._streak_start_s if converged else None,
            "time_to_stable_s": (
                self._streak_start_s - self.constant_start_s
                if converged and self._streak_start_s is not None else None
            ),
            "convergence_detected_s": self._convergence_detected_s,
            "stable_streak_count": self.stable_streak_count,
            "stable_window_count": self.stable_streak_count,
            "reset_count": self.reset_count,
            "steady_tow_force_n": tension,
            "plateau_force_n": tension,
            "all_wheels_i_clamped": all_i_clamped,
            "effort_clamp_any": self._effort_clamp_any,
            "per_wheel": per_wheel,
            "windows": list(self.windows),
            "last_window": self.windows[-1] if self.windows else None,
            "steady_state_config": self.config.to_dict(),
        }
