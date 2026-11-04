from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

from traction_tendon_steady import (  # noqa: E402
    AdaptiveSteadyMonitor,
    SteadyStateConfig,
    analyze_sustained_steady_state,
)


WHEELS = (
    "left_front_wheel_joint",
    "right_front_wheel_joint",
    "left_back_wheel_joint",
    "right_back_wheel_joint",
)


def config(**overrides):
    values = {
        "window_s": 2.0,
        "window_step_s": 0.25,
        "min_observation_s": 2.0,
        "required_stable_windows": 4,
        "max_wait_s": 14.0,
        "min_tendon_active_ratio": 0.95,
        "max_force_cv": 0.05,
        "max_abs_force_slope_n_s": 0.5,
        "max_constraint_extension_m": 0.005,
        "max_abs_base_velocity_mps": 0.005,
        "max_abs_i_term_slope_nm_s": 0.005,
        "min_i_clamp_ratio": 0.95,
    }
    values.update(overrides)
    return SteadyStateConfig(**values)


def synthetic_rows(
    *,
    axis="x",
    end_s=14.0,
    tension_at=lambda t: 140.0,
    i_term_at=lambda t: 2.0,
    i_clamped_at=lambda t: True,
    effort_clamped_at=lambda t: False,
    active_at=lambda t: True,
    velocity_at=lambda t: 0.0,
    extension_at=lambda t: 0.001,
):
    state = []
    odom = []
    pid = []
    index = 0
    while index * 0.05 <= end_s + 1e-9:
        timestamp = round(index * 0.05, 10)
        state.append({
            "t_experiment": timestamp,
            "tension_n": float(tension_at(timestamp)),
            "limit_active_flag": float(bool(active_at(timestamp))),
            "extension_over_upper_limit_m": float(extension_at(timestamp)),
        })
        odom.append({
            "t_experiment": timestamp,
            "vx": float(velocity_at(timestamp)) if axis == "x" else 0.0,
            "vy": float(velocity_at(timestamp)) if axis == "y" else 0.0,
        })
        for wheel in WHEELS:
            i_term = float(i_term_at(timestamp))
            pid.append({
                "t_experiment": timestamp,
                "joint_name": wheel,
                "i_term": i_term,
                "i_clamp_active": float(bool(i_clamped_at(timestamp))),
                "effort_clamp_active": float(bool(effort_clamped_at(timestamp))),
                "qfrc_applied": 10.0 + i_term,
            })
        index += 1
    return state, odom, pid


@pytest.mark.parametrize("axis", ["x", "y"])
def test_x_and_y_delayed_stability_uses_persistent_windows(axis):
    state, odom, pid = synthetic_rows(
        axis=axis,
        tension_at=lambda t: 100.0 + 5.0 * min(t, 8.0),
        i_term_at=lambda t: min(0.25 * t, 2.0),
        i_clamped_at=lambda t: t >= 8.0,
    )

    result = analyze_sustained_steady_state(
        state,
        odom,
        pid,
        axis=axis,
        constant_start_s=0.0,
        constant_end_s=14.0,
        config=config(),
        wheel_names=WHEELS,
    )

    assert result["converged"] is True
    assert result["status"] == "converged"
    assert result["time_to_stable_s"] >= 8.0
    assert result["steady_tow_force_n"] == pytest.approx(140.0)
    assert result["all_wheels_i_clamped"] is True
    assert result["effort_clamp_any"] is False


def test_low_speed_slow_integral_is_not_steady_even_with_low_force_cv():
    state, odom, pid = synthetic_rows(
        tension_at=lambda t: 120.0 + 0.2 * t,
        i_term_at=lambda t: 0.02 * t,
        i_clamped_at=lambda t: False,
    )

    result = analyze_sustained_steady_state(
        state,
        odom,
        pid,
        axis="x",
        constant_start_s=0.0,
        constant_end_s=14.0,
        config=config(),
        wheel_names=WHEELS,
    )

    assert result["converged"] is False
    assert result["status"] == "not_converged"
    assert "i_term_slope" in result["last_window"]["reasons"]


def test_continuously_climbing_tension_times_out_not_converged():
    state, odom, pid = synthetic_rows(
        tension_at=lambda t: 100.0 + 2.0 * t,
    )

    result = analyze_sustained_steady_state(
        state,
        odom,
        pid,
        axis="x",
        constant_start_s=0.0,
        constant_end_s=14.0,
        config=config(),
        wheel_names=WHEELS,
    )

    assert result["converged"] is False
    assert result["steady_tow_force_n"] is None
    assert "force_slope" in result["last_window"]["reasons"]


def test_short_false_plateau_is_reset_by_later_instability():
    def tension(timestamp):
        if timestamp < 3.0:
            return 100.0 + 3.0 * timestamp
        if timestamp < 6.0:
            return 109.0
        if timestamp < 9.0:
            return 109.0 + 10.0 * (timestamp - 6.0)
        return 139.0

    cfg = config(required_stable_windows=10, max_wait_s=14.0)
    state, odom, pid = synthetic_rows(tension_at=tension)
    monitor = AdaptiveSteadyMonitor(
        axis="x", constant_start_s=0.0, config=cfg, wheel_names=WHEELS
    )
    saw_early_streak = False
    saw_reset = False
    for row in state:
        timestamp = row["t_experiment"]
        monitor.add_state(row)
        monitor.add_odom(odom[len(monitor.state_history) - 1])
        pid_start = (len(monitor.state_history) - 1) * len(WHEELS)
        for pid_row in pid[pid_start:pid_start + len(WHEELS)]:
            monitor.add_pid(pid_row)
        decision = monitor.advance(timestamp)
        saw_early_streak = saw_early_streak or decision["stable_streak_count"] > 0
        saw_reset = saw_reset or decision["reset_count"] > 0

    assert saw_early_streak is True
    assert saw_reset is True
    assert monitor.result()["status"] == "converged"
    assert monitor.result()["stable_start_s"] >= 8.5


def test_effort_clamp_is_a_hard_invalid_result():
    state, odom, pid = synthetic_rows(
        effort_clamped_at=lambda t: t >= 5.0,
    )

    result = analyze_sustained_steady_state(
        state,
        odom,
        pid,
        axis="x",
        constant_start_s=0.0,
        constant_end_s=14.0,
        config=config(),
        wheel_names=WHEELS,
    )

    assert result["effort_clamp_any"] is True
    assert result["valid"] is False
    assert result["status"] == "effort_clamp"
    assert "effort_clamp" in result["reject_reasons"]


def test_online_monitor_reports_timeout_instead_of_using_last_window():
    cfg = config(max_wait_s=5.0)
    state, odom, pid = synthetic_rows(
        end_s=5.0,
        tension_at=lambda t: 100.0 + 2.0 * t,
    )
    monitor = AdaptiveSteadyMonitor(
        axis="x", constant_start_s=0.0, config=cfg, wheel_names=WHEELS
    )
    decision = None
    for index, row in enumerate(state):
        monitor.add_state(row)
        monitor.add_odom(odom[index])
        for pid_row in pid[index * 4:index * 4 + 4]:
            monitor.add_pid(pid_row)
        decision = monitor.advance(row["t_experiment"])

    assert decision["status"] == "not_converged"
    assert decision["converged"] is False
    assert decision["steady_tow_force_n"] is None


def test_production_contract_contains_no_effect_repeatability_or_monotonic_gate():
    keys = set(config().to_dict())
    forbidden = ("effect", "repeat", "monotonic", "three_percent", "3_percent")
    assert not any(token in key for token in forbidden for key in keys)
    assert math.isclose(config().max_wait_s, 14.0)
