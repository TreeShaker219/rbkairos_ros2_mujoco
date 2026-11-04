from __future__ import annotations

from collections import Counter
import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from evaluation.config import EvaluationConfig
from evaluation.costs import compute_multispeed_total_cost
from evaluation.evaluators import (
    FrfMetric,
    TractionMetric,
    evaluate_frf_offline,
)
from evaluation.metadata_validation import validate_runtime_metadata_multi
from evaluation.parameters import EvaluationParameters
from evaluation.result import (
    EvaluationResult,
    _create_runtime_assets_from_config,
    _evaluate_traction_metric,
    _git_commit_or_none,
    _make_output_dir,
    _write_result,
)
from evaluation.runners import (
    ExperimentRunRef,
    build_frf_command_from_config,
    build_traction_command_from_config,
    frf_timeout_from_config,
    latest_frequency_response_experiment,
    latest_traction_tendon_run,
    run_command_ref,
    traction_timeout_from_config,
)
from evaluation.runtime_assets import RuntimeAssets
from evaluation.simulation_session import SimulationSession, build_default_launch_command
from evaluation.traction_conditions import (
    TRACTION_CONDITION_IDS,
    condition_axis,
)
from evaluation.frf_reuse import frf_task_id


MANAGED_BLOCKING_ROS_NODES = (
    "/robot_state_publisher",
    "/mujoco_sim_ros2_node",
    "/controller_manager",
)

MANAGED_REQUIRED_ACTIVE_CONTROLLERS = (
    "joint_state_broadcaster",
    "mecanum_base_controller",
)

CONTROLLER_READINESS_SCRIPT = r"""
import json
import os
import sys
import time

payload = json.loads(sys.argv[1])
service_name = payload["service_name"]
timeout_s = max(float(payload["timeout_s"]), 0.1)
required_active = list(payload.get("required_active", []))
started = time.monotonic()
result = {
    "status": "unknown",
    "service": service_name,
    "controllers": {},
    "missing": [],
    "inactive": {},
    "error": None,
}

try:
    import rclpy
    from controller_manager_msgs.srv import ListControllers
    from rclpy.signals import SignalHandlerOptions
except Exception as exc:
    result.update({
        "status": "error",
        "error": f"controller readiness imports failed: {exc}",
    })
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0)

node = None
try:
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node(
        f"rbkairos_controller_readiness_{os.getpid()}",
        start_parameter_services=False,
    )
    client = node.create_client(ListControllers, service_name)
    service_deadline = time.monotonic() + timeout_s
    while not client.service_is_ready():
        if time.monotonic() >= service_deadline:
            result.update({
                "status": "error",
                "error": f"controller service unavailable: {service_name}",
            })
            print(json.dumps(result, sort_keys=True))
            raise SystemExit(0)
        client.wait_for_service(timeout_sec=0.05)

    future = client.call_async(ListControllers.Request())
    call_deadline = time.monotonic() + timeout_s
    while not future.done():
        if time.monotonic() >= call_deadline:
            result.update({
                "status": "error",
                "error": f"controller service call timed out: {service_name}",
            })
            print(json.dumps(result, sort_keys=True))
            raise SystemExit(0)
        rclpy.spin_once(node, timeout_sec=0.05)

    exception = future.exception()
    if exception is not None:
        result.update({
            "status": "error",
            "error": f"controller service call failed: {exception}",
        })
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(0)

    response = future.result()
    controllers = {
        controller.name: {
            "state": controller.state,
            "type": controller.type,
            "line": f"{controller.name} {controller.type} {controller.state}",
        }
        for controller in response.controller
    }
    missing = [
        controller
        for controller in required_active
        if controller not in controllers
    ]
    inactive = {
        controller: controllers[controller].get("state", "")
        for controller in required_active
        if controller in controllers and controllers[controller].get("state") != "active"
    }
    status = "ready"
    error = None
    if missing or inactive:
        status = "not_ready"
        detail_parts = []
        if missing:
            detail_parts.append("missing controllers: " + ", ".join(missing))
        if inactive:
            detail_parts.append(
                "inactive controllers: "
                + ", ".join(f"{name}={state}" for name, state in inactive.items())
            )
        error = "; ".join(detail_parts)
    result.update({
        "status": status,
        "controllers": controllers,
        "missing": missing,
        "inactive": inactive,
        "error": error,
    })
except Exception as exc:
    result.update({
        "status": "error",
        "error": f"controller readiness check failed: {exc}",
    })
finally:
    if node is not None:
        try:
            node.destroy_node()
        except Exception:
            pass
    try:
        rclpy.shutdown()
    except Exception:
        pass
    result["duration_s"] = time.monotonic() - started
    print(json.dumps(result, sort_keys=True))
"""


def _managed_float(config: dict[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)


def _managed_int(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if value is None or str(value).strip() == "":
        return int(default)
    return int(value)


def _managed_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _managed_string_list(config: dict[str, Any], key: str, default: tuple[str, ...]) -> list[str]:
    value = config.get(key)
    if value is None or value == "":
        return list(default)
    if isinstance(value, str):
        return [item for item in value.replace(",", " ").split() if item]
    return [str(item) for item in value if str(item).strip()]


def _ros_graph_checks_enabled(managed_config: dict[str, Any]) -> bool:
    return _managed_bool(managed_config, "ros_graph_checks", True)


def _ros_graph_snapshot(
    label: str,
    managed_config: dict[str, Any],
) -> dict[str, Any]:
    blocking_nodes = _managed_string_list(
        managed_config,
        "ros_graph_blocking_nodes",
        MANAGED_BLOCKING_ROS_NODES,
    )
    timeout_s = _managed_float(managed_config, "ros_graph_node_list_timeout_s", 1.0)
    use_daemon = _managed_bool(managed_config, "ros_graph_use_daemon", False)
    spin_time_s = max(0.0, _managed_float(managed_config, "ros_graph_spin_time_s", 0.1))
    command = ["ros2", "node", "list"]
    if not use_daemon:
        command.append("--no-daemon")
        command.extend(["--spin-time", f"{spin_time_s:.3f}"])
    ros_log_dir = str(managed_config.get("ros_graph_log_dir", "/tmp/rbkairos_ros_graph_checks"))
    env = os.environ.copy()
    if ros_log_dir:
        try:
            Path(ros_log_dir).mkdir(parents=True, exist_ok=True)
            env["ROS_LOG_DIR"] = ros_log_dir
        except OSError:
            pass
    started = time.monotonic()
    snapshot: dict[str, Any] = {
        "label": label,
        "status": "unknown",
        "command": command,
        "use_daemon": use_daemon,
        "spin_time_s": spin_time_s,
        "ros_log_dir": env.get("ROS_LOG_DIR"),
        "blocking_nodes": blocking_nodes,
        "blocking_counts": {},
        "duplicate_counts": {},
        "node_count": 0,
        "nodes": [],
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "error": None,
        "duration_s": None,
    }
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(timeout_s, 0.1),
            check=False,
            env=env,
        )
    except FileNotFoundError as exc:
        snapshot.update({
            "status": "error",
            "error": f"ros2 not found: {exc}",
            "duration_s": time.monotonic() - started,
        })
        return snapshot
    except subprocess.TimeoutExpired as exc:
        snapshot.update({
            "status": "error",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "error": f"ros2 node list timed out after {timeout_s:.3f}s",
            "duration_s": time.monotonic() - started,
        })
        return snapshot

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    nodes = [
        line.strip()
        for line in stdout.splitlines()
        if line.strip().startswith("/")
    ]
    counts = Counter(nodes)
    blocking_counts = {
        node: counts[node]
        for node in blocking_nodes
        if counts.get(node, 0) > 0
    }
    duplicate_counts = {
        node: count
        for node, count in counts.items()
        if count > 1
    }
    status = "clean"
    error = None
    if completed.returncode != 0:
        status = "error"
        error = f"ros2 node list exited with {completed.returncode}"
    elif blocking_counts:
        status = "dirty"
        error = "blocking ROS nodes are already present"

    snapshot.update({
        "status": status,
        "blocking_counts": blocking_counts,
        "duplicate_counts": duplicate_counts,
        "node_count": len(nodes),
        "nodes": nodes,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "error": error,
        "duration_s": time.monotonic() - started,
    })
    return snapshot


def _wait_for_clean_ros_graph(
    label: str,
    managed_config: dict[str, Any],
    wait_key: str,
) -> dict[str, Any]:
    if not _ros_graph_checks_enabled(managed_config):
        return {
            "label": label,
            "status": "skipped",
            "waited_s": 0.0,
            "attempts": 0,
            "snapshot": None,
        }

    default_wait_s = {
        "ros_graph_preflight_wait_s": 0.5,
        "ros_graph_cleanup_wait_s": 5.0,
        "ros_graph_force_kill_wait_s": 1.0,
    }.get(wait_key, 0.0)
    wait_s = max(0.0, _managed_float(managed_config, wait_key, default_wait_s))
    poll_s = max(0.05, _managed_float(managed_config, "ros_graph_poll_s", 0.25))
    started = time.monotonic()
    deadline = started + wait_s
    attempts = 0
    last_snapshot: dict[str, Any] | None = None

    while True:
        attempts += 1
        last_snapshot = _ros_graph_snapshot(label, managed_config)
        if last_snapshot["status"] in {"clean", "error"}:
            return {
                "label": label,
                "status": last_snapshot["status"],
                "waited_s": time.monotonic() - started,
                "attempts": attempts,
                "snapshot": last_snapshot,
            }

        now = time.monotonic()
        if now >= deadline:
            return {
                "label": label,
                "status": last_snapshot["status"],
                "waited_s": now - started,
                "attempts": attempts,
                "snapshot": last_snapshot,
            }
        time.sleep(min(poll_s, max(0.0, deadline - now)))


def _ros_graph_check_ok(check: dict[str, Any]) -> bool:
    return check.get("status") in {"clean", "skipped"}


def _ros_graph_error_text(check: dict[str, Any]) -> str:
    snapshot = check.get("snapshot")
    if isinstance(snapshot, dict) and snapshot.get("error"):
        return str(snapshot["error"])
    return str(check.get("status", "unknown"))


def _controller_readiness_checks_enabled(managed_config: dict[str, Any]) -> bool:
    return _managed_bool(managed_config, "controller_readiness_checks", True)


def _controller_readiness_snapshot(
    label: str,
    managed_config: dict[str, Any],
) -> dict[str, Any]:
    required_active = _managed_string_list(
        managed_config,
        "controller_required_active",
        MANAGED_REQUIRED_ACTIVE_CONTROLLERS,
    )
    timeout_s = _managed_float(managed_config, "controller_list_timeout_s", 2.0)
    controller_manager = str(
        managed_config.get("controller_manager_node", "/controller_manager")
    ).strip() or "/controller_manager"
    service_name = f"{controller_manager.rstrip('/')}/list_controllers"
    ros_log_dir = str(
        managed_config.get(
            "controller_log_dir",
            managed_config.get("ros_graph_log_dir", "/tmp/rbkairos_ros_graph_checks"),
        )
    )
    env = os.environ.copy()
    if ros_log_dir:
        try:
            Path(ros_log_dir).mkdir(parents=True, exist_ok=True)
            env["ROS_LOG_DIR"] = ros_log_dir
        except OSError:
            pass

    started = time.monotonic()
    payload = {
        "service_name": service_name,
        "timeout_s": timeout_s,
        "required_active": required_active,
    }
    command = [
        sys.executable,
        "-c",
        CONTROLLER_READINESS_SCRIPT,
        json.dumps(payload, sort_keys=True),
    ]
    snapshot: dict[str, Any] = {
        "label": label,
        "status": "unknown",
        "method": "rclpy_service_subprocess",
        "service": service_name,
        "command": [sys.executable, "-c", "<controller_readiness_script>", "<payload>"],
        "controller_manager": controller_manager,
        "required_active": required_active,
        "controllers": {},
        "missing": [],
        "inactive": {},
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "error": None,
        "duration_s": None,
        "ros_log_dir": env.get("ROS_LOG_DIR"),
    }

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(timeout_s * 2.0 + 1.0, 1.0),
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        snapshot.update({
            "status": "error",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "error": f"controller readiness subprocess timed out after {timeout_s:.3f}s",
            "duration_s": time.monotonic() - started,
        })
        return snapshot
    except OSError as exc:
        snapshot.update({
            "status": "error",
            "error": f"controller readiness subprocess failed: {exc}",
            "duration_s": time.monotonic() - started,
        })
        return snapshot

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    result = None
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                result = json.loads(stripped)
            except json.JSONDecodeError:
                pass
            break
    if not isinstance(result, dict):
        snapshot.update({
            "status": "error",
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "error": "controller readiness subprocess did not return JSON",
            "duration_s": time.monotonic() - started,
        })
        return snapshot

    status = str(result.get("status", "error"))
    snapshot.update({
        "status": status,
        "controllers": result.get("controllers", {}),
        "missing": result.get("missing", []),
        "inactive": result.get("inactive", {}),
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "error": result.get("error"),
        "duration_s": time.monotonic() - started,
    })
    if completed.returncode != 0 and status != "ready":
        snapshot["status"] = "error"
        if not snapshot.get("error"):
            snapshot["error"] = (
                f"controller readiness subprocess exited with {completed.returncode}"
            )
    return snapshot


def _wait_for_active_controllers(
    label: str,
    managed_config: dict[str, Any],
) -> dict[str, Any]:
    if not _controller_readiness_checks_enabled(managed_config):
        return {
            "label": label,
            "status": "skipped",
            "waited_s": 0.0,
            "attempts": 0,
            "snapshot": None,
        }

    wait_s = max(
        0.0,
        _managed_float(managed_config, "controller_readiness_wait_s", 3.0),
    )
    poll_s = max(
        0.05,
        _managed_float(managed_config, "controller_readiness_poll_s", 0.25),
    )
    started = time.monotonic()
    deadline = started + wait_s
    attempts = 0
    last_snapshot: dict[str, Any] | None = None

    while True:
        attempts += 1
        last_snapshot = _controller_readiness_snapshot(label, managed_config)
        if last_snapshot["status"] == "ready":
            return {
                "label": label,
                "status": last_snapshot["status"],
                "waited_s": time.monotonic() - started,
                "attempts": attempts,
                "snapshot": last_snapshot,
            }

        now = time.monotonic()
        if now >= deadline:
            return {
                "label": label,
                "status": last_snapshot["status"],
                "waited_s": now - started,
                "attempts": attempts,
                "snapshot": last_snapshot,
            }
        time.sleep(min(poll_s, max(0.0, deadline - now)))


def _controller_check_ok(check: dict[str, Any]) -> bool:
    return check.get("status") in {"ready", "skipped"}


def _controller_error_text(check: dict[str, Any]) -> str:
    snapshot = check.get("snapshot")
    if isinstance(snapshot, dict) and snapshot.get("error"):
        return str(snapshot["error"])
    return str(check.get("status", "unknown"))


def _frf_frequencies(config: dict[str, Any]) -> list[float]:
    raw = config.get("frequencies")
    if raw is None or raw == "":
        return [0.05, 0.10, 0.20, 0.50, 0.80, 1.20, 2.00]
    if isinstance(raw, str):
        values = raw.replace(",", " ").split()
    else:
        values = list(raw)
    frequencies = [float(value) for value in values]
    if any(not math.isfinite(frequency) or frequency <= 0.0 for frequency in frequencies):
        raise ValueError("FRF frequencies must be finite positive values")
    return frequencies


def _frequency_run_id(frequency_hz: float) -> str:
    return f"f_{float(frequency_hz):.3f}".replace(".", "p") + "Hz"


def _single_frequency_frf_config(config: dict[str, Any], frequency_hz: float) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["frequencies"] = [float(frequency_hz)]
    return result


def _single_frequency_run_dir(
    source_experiment_dir: Path,
    frequency_hz: float,
) -> Path | None:
    runs_dir = source_experiment_dir / "runs"
    if not runs_dir.is_dir():
        return None

    expected_run_dir = runs_dir / _frequency_run_id(frequency_hz)
    if expected_run_dir.is_dir():
        source_run_dir = expected_run_dir
    else:
        candidates = sorted(path for path in runs_dir.iterdir() if path.is_dir())
        if len(candidates) != 1:
            return None
        source_run_dir = candidates[0]

    return source_run_dir


def _write_per_frequency_frf_metadata(
    per_frequency_root: Path,
    frf_config: dict[str, Any],
    frequency_results: list[dict[str, Any]],
) -> Path:
    collected_frequencies = [
        result["frequency_hz"]
        for result in frequency_results
        if result.get("source_run_dir") is not None
    ]
    metadata = {
        "metadata_schema_version": 1,
        "experiment_type": "frequency_response",
        "collection_method": "managed_per_frequency_mujoco_sessions",
        "analysis_method": "direct_recursive_per_frequency",
        "created_at_epoch": time.time(),
        "frequency_list": collected_frequencies,
        "script_parameters": copy.deepcopy(frf_config),
        "analysis_input_dir": str(per_frequency_root),
        "runs_layout": "per_frequency/<frequency_id>/<experiment_dir>/runs/<run_id>",
        "source_frequency_results": frequency_results,
    }
    per_frequency_root.mkdir(parents=True, exist_ok=True)
    metadata_path = per_frequency_root / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    return metadata_path


def _base_provenance(
    mode: str,
    config: EvaluationConfig,
    assets: RuntimeAssets | None,
    started: float,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "git_commit": _git_commit_or_none(Path.cwd()),
        "config_snapshot": config.to_dict(),
        "runtime_assets": assets.to_dict() if assets else None,
        "cache_decision": {
            "reuse_policy": config.reuse_policy,
            "used_cache": False,
            "reason": f"fresh_{mode}_evaluation",
        },
        "wall_time": {
            "started_at_epoch": started,
            "duration_s": time.time() - started,
        },
        "metadata_paths": {},
        "runtime_metadata_validation": None,
        "launch_commands": [],
        "analyzer_commands": [],
    }


def _run_frf_analysis(
    experiment_dir: Path,
    output_dir: Path,
    analyzer_config: dict[str, Any],
) -> dict[str, Any]:
    import analyze_frequency_response_fft as frf_analysis

    return frf_analysis.analyze_frequency_response_fft(
        experiment_dir,
        output_dir=output_dir,
        environment=str(analyzer_config.get("environment", "auto")),
        output_source=str(analyzer_config.get("output_source", "auto")),
        discard_cycles=_managed_float(
            analyzer_config,
            "discard_cycles",
            _managed_float(analyzer_config, "default_discard_cycles", 2.0),
        ),
        ramp_cycles=_managed_float(analyzer_config, "ramp_cycles", 1.0),
        analysis_cycles=_managed_int(analyzer_config, "analysis_cycles", 4),
        min_analysis_cycles=_managed_float(analyzer_config, "min_analysis_cycles", 2.0),
        grid_rate_hz=_managed_float(
            analyzer_config,
            "grid_rate_hz",
            _managed_float(analyzer_config, "resample_rate", 50.0),
        ),
        cycle_shortfall_tolerance=_managed_float(
            analyzer_config,
            "cycle_shortfall_tolerance",
            0.05,
        ),
        max_harmonic=_managed_int(analyzer_config, "max_harmonic", 7),
        nyquist_safety=_managed_float(analyzer_config, "nyquist_safety", 0.90),
        plots=_managed_bool(analyzer_config, "plots", True),
        html=_managed_bool(analyzer_config, "html", True),
        html_output=analyzer_config.get("html_output"),
        max_spectrum_plots=_managed_int(analyzer_config, "max_spectrum_plots", 8),
    )


def _launch_command_for_assets(
    assets: RuntimeAssets,
    model_path: Path,
    traction_config: dict[str, Any],
    traction_tendon_enabled: bool,
) -> list[str]:
    if assets.urdf_path is None:
        raise ValueError("runtime asset missing urdf_path")
    if assets.config_path is None:
        raise ValueError("runtime asset missing config_path")
    return build_default_launch_command(
        model_file=str(model_path.resolve()),
        urdf_file=str(assets.urdf_path.resolve()),
        config_file=str(assets.config_path.resolve()),
        model_package="rbkairos",
        traction_tendon_enabled=traction_tendon_enabled,
        traction_tendon_name=str(
            traction_config.get("tendon_name", "traction_cable_x")
        ),
        traction_tendon_sensor_name=str(
            traction_config.get("sensor_name", "traction_cable_x_limit_force")
        ),
        traction_tendon_attach_site_name=str(
            traction_config.get("attach_site_name", "traction_attach_x")
        ),
        traction_tendon_anchor_site_name=str(
            traction_config.get("anchor_site_name", "traction_anchor_x")
        ),
        traction_tendon_publish_rate_hz=_managed_float(
            traction_config,
            "observer_publish_rate",
            100.0,
        ),
        traction_tendon_frame_id=str(
            traction_config.get("attach_site_name", "traction_attach_x")
        ),
    )


def build_managed_dry_run(
    config: EvaluationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Build the canonical managed sequence without launching or writing files."""
    asset_dir = output_dir / "runtime_assets"
    base_model = asset_dir / "rbkairos.parameterized.xml"
    urdf_path = asset_dir / "rbkairos.parameterized.urdf"
    config_path = asset_dir / "mujoco_mecanum_drive.parameterized.yaml"
    steps: list[dict[str, Any]] = []
    for condition_id in TRACTION_CONDITION_IDS:
        axis = condition_axis(condition_id)
        traction_config = copy.deepcopy(
            config.traction_experiments.get(condition_id, {})
        )
        traction_config.setdefault("axis", axis)
        scene = asset_dir / f"rbkairos_scene_tendon_test_{axis}.parameterized.xml"
        launch = build_default_launch_command(
            model_file=str(scene),
            urdf_file=str(urdf_path),
            config_file=str(config_path),
            traction_tendon_enabled=True,
            traction_tendon_name=str(
                traction_config.get("tendon_name", f"traction_cable_{axis}")
            ),
            traction_tendon_sensor_name=str(
                traction_config.get(
                    "sensor_name", f"traction_cable_{axis}_limit_force"
                )
            ),
            traction_tendon_attach_site_name=str(
                traction_config.get("attach_site_name", f"traction_attach_{axis}")
            ),
            traction_tendon_anchor_site_name=str(
                traction_config.get("anchor_site_name", f"traction_anchor_{axis}")
            ),
            traction_tendon_frame_id=str(
                traction_config.get("attach_site_name", f"traction_attach_{axis}")
            ),
        )
        traction_config["model_file"] = str(scene)
        traction_config["urdf_file"] = str(urdf_path)
        output_base = output_dir / str(
            config.managed.get("traction_output_dirs", {}).get(
                condition_id, f"traction/{condition_id}"
            )
        )
        steps.append(
            {
                "phase": f"traction_{condition_id}",
                "launch_command": launch,
                "recorder_command": build_traction_command_from_config(
                    traction_config,
                    output_base,
                ),
            }
        )
        steps.append({"phase": f"cleanup_{condition_id}"})

    for frequency_hz in _frf_frequencies(config.frf_experiment):
        frequency_label = f"{float(frequency_hz):.9f}".replace(".", "p")
        launch = build_default_launch_command(
            model_file=str(base_model),
            urdf_file=str(urdf_path),
            config_file=str(config_path),
            traction_tendon_enabled=False,
        )
        frf_config = _single_frequency_frf_config(
            config.frf_experiment,
            frequency_hz,
        )
        steps.append(
            {
                "phase": f"frf_{frequency_label}hz",
                "launch_command": launch,
                "recorder_command": build_frf_command_from_config(
                    frf_config,
                    output_dir / "frf" / "per_frequency" / frequency_label,
                ),
            }
        )
        steps.append({"phase": f"cleanup_frf_{frequency_label}hz"})
    return {"steps": steps}


def _stop_session_and_check_ros_graph(
    session: SimulationSession,
    label: str,
    managed_config: dict[str, Any],
    shutdown_timeout_s: float,
    ros_graph_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    session.stop(timeout_s=shutdown_timeout_s)
    cleanup_check = _wait_for_clean_ros_graph(
        f"{label}.post_stop",
        managed_config,
        "ros_graph_cleanup_wait_s",
    )
    ros_graph_checks.append(cleanup_check)
    if _ros_graph_check_ok(cleanup_check) or cleanup_check.get("status") != "dirty":
        return cleanup_check

    session.force_kill(
        timeout_s=max(0.1, _managed_float(managed_config, "force_kill_timeout_s", 1.0))
    )
    force_check = _wait_for_clean_ros_graph(
        f"{label}.post_force_kill",
        managed_config,
        "ros_graph_force_kill_wait_s",
    )
    ros_graph_checks.append(force_check)
    return force_check


def evaluate_managed(
    parameters: EvaluationParameters,
    config: EvaluationConfig,
    output_dir: Path | None = None,
    create_assets: bool = True,
    strict: bool = False,
    resume_task_outputs: dict[str, str] | None = None,
    frf_reuse_bundle: dict[str, Any] | None = None,
    task_callback: Callable[[str, str, dict[str, Any]], None] | None = None,
    stop_on_task_failure: bool = False,
    persist_result: bool = True,
    experiment_scope: str = "all",
) -> EvaluationResult:
    started = time.time()
    parameters.validate()
    if experiment_scope not in {"all", "tendon", "frf"}:
        raise ValueError("experiment_scope must be one of all, tendon, frf")
    collect_tendon = experiment_scope in {"all", "tendon"}
    collect_frf = experiment_scope in {"all", "frf"}
    output_dir = output_dir or _make_output_dir(config, parameters)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_task_outputs = dict(resume_task_outputs or {})

    assets: RuntimeAssets | None = None
    if create_assets:
        assets = _create_runtime_assets_from_config(
            parameters,
            config,
            output_dir / "runtime_assets",
        )

    traction_metrics: dict[str, TractionMetric] = {}
    frf_metric: FrfMetric | None = None
    traction_run_dirs: dict[str, Path | None] = {
        condition_id: None for condition_id in TRACTION_CONDITION_IDS
    }
    frf_summary_path: Path | None = None
    traction_sessions: dict[str, SimulationSession] = {}
    traction_run_refs: dict[str, ExperimentRunRef] = {}
    frf_sessions: list[SimulationSession] = []
    frf_run_refs: list[ExperimentRunRef] = []
    frf_frequency_results: list[dict[str, Any]] = []
    frf_single_output_root: Path | None = None
    frf_analysis_input_dir: Path | None = None
    frf_per_frequency_metadata_path: Path | None = None
    frf_analysis_result: dict[str, Any] | None = None
    frf_reused_from: str | None = None
    frf_nonbinding_certificate: dict[str, Any] | None = None
    frf_source_raw_data_paths: list[str] = []
    managed_errors: dict[str, str] = {}
    ros_graph_checks: list[dict[str, Any]] = []
    controller_checks: list[dict[str, Any]] = []
    managed_runtime_blocked = False
    reject_reasons: list[str] = []

    if frf_reuse_bundle is not None:
        source_id = frf_reuse_bundle.get("source_candidate_id")
        certificate = frf_reuse_bundle.get("certificate")
        reused_metric = frf_reuse_bundle.get("frf_metric")
        raw_paths = frf_reuse_bundle.get("source_raw_data_paths")
        if (
            not isinstance(source_id, str)
            or not source_id
            or not isinstance(certificate, dict)
            or not isinstance(reused_metric, FrfMetric)
            or not reused_metric.valid
            or not isinstance(raw_paths, dict)
            or not raw_paths
        ):
            raise ValueError("invalid managed FRF reuse bundle")
        frf_reused_from = source_id
        frf_nonbinding_certificate = certificate
        frf_source_raw_data_paths = [str(path) for path in raw_paths.values()]
        frf_metric = reused_metric

    if assets is None:
        reject_reasons.append("managed.runtime_assets_missing")
    elif assets.mujoco_model_path is None:
        reject_reasons.append("managed.mujoco_model_missing")
    elif set(assets.traction_mujoco_model_paths) != {"x", "y"}:
        reject_reasons.append("managed.traction_mujoco_models_missing")
    traction_real_forces = config.traction_real_forces
    for condition_id, real_force in traction_real_forces.items():
        if real_force is None:
            reject_reasons.append(
                f"missing_references.traction_real_forces_n.{condition_id}"
            )
    if config.frf_real_reference is None:
        reject_reasons.append("missing_references.frf_real_reference")
    traction_experiments = config.traction_experiments
    for condition_id in TRACTION_CONDITION_IDS:
        if condition_id not in traction_experiments:
            reject_reasons.append(
                f"managed.traction_{condition_id}_experiment_missing"
            )
    can_run_experiments = (
        assets is not None
        and assets.mujoco_model_path is not None
        and set(assets.traction_mujoco_model_paths) == {"x", "y"}
        and all(value is not None for value in traction_real_forces.values())
        and config.frf_real_reference is not None
        and all(
            condition_id in traction_experiments
            for condition_id in TRACTION_CONDITION_IDS
        )
    )
    if not can_run_experiments:
        managed_errors["skipped"] = (
            "managed execution skipped because required assets or references are missing"
        )

    managed_config = config.managed
    default_traction_config = traction_experiments.get(
        TRACTION_CONDITION_IDS[0], {}
    )
    frf_config = config.frf_experiment
    logs_dir = output_dir / str(managed_config.get("logs_dir", "logs"))
    traction_output_dirs_config = managed_config.get("traction_output_dirs", {})
    if not isinstance(traction_output_dirs_config, dict):
        traction_output_dirs_config = {}
    traction_output_bases = {
        condition_id: output_dir
        / str(
            traction_output_dirs_config.get(
                condition_id,
                f"traction/{condition_id}",
            )
        )
        for condition_id in TRACTION_CONDITION_IDS
    }
    frf_output_base = output_dir / str(managed_config.get("frf_output_dir", "frf"))
    frf_analysis_dir = output_dir / str(
        managed_config.get("frf_analysis_dir", "frf/analysis")
    )
    startup_wait_s = _managed_float(managed_config, "startup_wait_s", 8.0)
    shutdown_timeout_s = _managed_float(managed_config, "shutdown_timeout_s", 10.0)
    command_timeout_margin_s = _managed_float(
        managed_config,
        "command_timeout_margin_s",
        30.0,
    )

    if can_run_experiments:
        preflight_check = _wait_for_clean_ros_graph(
            "managed.preflight",
            managed_config,
            "ros_graph_preflight_wait_s",
        )
        ros_graph_checks.append(preflight_check)
        if not _ros_graph_check_ok(preflight_check):
            managed_runtime_blocked = True
            managed_errors["ros_graph_preflight"] = _ros_graph_error_text(preflight_check)
            reject_reasons.append("managed.ros_graph_preflight_failed")

    if (
        can_run_experiments
        and not managed_runtime_blocked
        and assets is not None
        and set(assets.traction_mujoco_model_paths) == {"x", "y"}
    ):
        analyzer_config = config.traction_analyzer
        for condition_id in TRACTION_CONDITION_IDS:
            task_id = f"traction_{condition_id}"
            axis = condition_axis(condition_id)
            traction_config = copy.deepcopy(traction_experiments[condition_id])
            traction_config.setdefault("axis", axis)
            traction_model_path = assets.traction_mujoco_model_paths[axis]
            traction_config["model_file"] = str(traction_model_path.resolve())
            if assets.urdf_path is not None:
                traction_config["urdf_file"] = str(assets.urdf_path.resolve())
            traction_output_base = traction_output_bases[condition_id]
            traction_session: SimulationSession | None = None
            resumed_run = resume_task_outputs.get(task_id)
            if resumed_run is not None:
                traction_run_dir = Path(resumed_run)
                traction_run_dirs[condition_id] = traction_run_dir
                try:
                    traction_metrics[condition_id] = _evaluate_traction_metric(
                        traction_run_dir,
                        float(traction_real_forces[condition_id]),
                        axis,
                        analyzer_config,
                        strict=bool(strict),
                    )
                    if not traction_metrics[condition_id].valid:
                        reject_reasons.append(
                            f"traction_{condition_id}:"
                            f"{traction_metrics[condition_id].reject_reason}"
                        )
                except Exception as exc:
                    managed_errors[f"traction_{condition_id}_resume_evaluator"] = str(exc)
                    reject_reasons.append(
                        f"managed.traction_{condition_id}_evaluator_failed"
                    )
                continue

            if not collect_tendon:
                continue

            if task_callback is not None:
                task_callback(task_id, "started", {})
            task_reasons_start = len(reject_reasons)
            task_status = "failed"
            task_last_error: str | None = None

            preflight_check = _wait_for_clean_ros_graph(
                f"traction_{condition_id}.preflight",
                managed_config,
                "ros_graph_preflight_wait_s",
            )
            ros_graph_checks.append(preflight_check)
            if not _ros_graph_check_ok(preflight_check):
                managed_runtime_blocked = True
                task_last_error = _ros_graph_error_text(preflight_check)
                managed_errors[f"ros_graph_preflight_traction_{condition_id}"] = (
                    task_last_error
                )
                reject_reasons.append("managed.ros_graph_preflight_failed")
                if task_callback is not None:
                    task_callback(
                        task_id,
                        "finished",
                        {
                            "status": "failed",
                            "output_dir": str(traction_output_base),
                            "raw_data_path": None,
                            "last_error": task_last_error,
                        },
                    )
                break

            try:
                traction_launch_command = _launch_command_for_assets(
                    assets,
                    traction_model_path,
                    traction_config,
                    traction_tendon_enabled=True,
                )
                traction_session = SimulationSession(
                    traction_launch_command,
                    logs_dir / f"traction_{condition_id}_mujoco",
                )
                traction_sessions[condition_id] = traction_session
                traction_session.start()
                time.sleep(startup_wait_s)
                if not traction_session.running():
                    reject_reasons.append(
                        f"managed.traction_{condition_id}_launch_exited"
                    )
                else:
                    traction_controller_check = _wait_for_active_controllers(
                        f"traction_{condition_id}.controllers_ready",
                        managed_config,
                    )
                    controller_checks.append(traction_controller_check)
                    if not _controller_check_ok(traction_controller_check):
                        managed_runtime_blocked = True
                        managed_errors[
                            f"controller_readiness_traction_{condition_id}"
                        ] = (
                            _controller_error_text(traction_controller_check)
                        )
                        reject_reasons.append("managed.controller_readiness_failed")
                    else:
                        traction_output_base.mkdir(parents=True, exist_ok=True)
                        traction_command = build_traction_command_from_config(
                            traction_config,
                            traction_output_base,
                        )
                        traction_run_ref = run_command_ref(
                            f"traction_{condition_id}_recorder",
                            traction_command,
                            traction_output_base,
                            logs_dir / "traction_recorders",
                            execute=True,
                            timeout_s=traction_timeout_from_config(
                                traction_config,
                                command_timeout_margin_s,
                            ),
                        )
                        traction_run_refs[condition_id] = traction_run_ref
                        if traction_run_ref.timed_out:
                            reject_reasons.append(
                                f"managed.traction_{condition_id}_recorder_timeout"
                            )
                        elif traction_run_ref.returncode not in (0, None):
                            reject_reasons.append(
                                f"managed.traction_{condition_id}_recorder_failed"
                            )
                        traction_run_dir = latest_traction_tendon_run(traction_output_base)
                        traction_run_dirs[condition_id] = traction_run_dir
                        traction_run_ref.discovered_output = traction_run_dir
                        if traction_run_dir is None:
                            reject_reasons.append(
                                f"managed.traction_{condition_id}_output_missing"
                            )
                        elif traction_real_forces[condition_id] is not None:
                            try:
                                traction_metrics[condition_id] = _evaluate_traction_metric(
                                    traction_run_dir,
                                    float(traction_real_forces[condition_id]),
                                    axis,
                                    analyzer_config,
                                    strict=bool(strict),
                                )
                                if not traction_metrics[condition_id].valid:
                                    reject_reasons.append(
                                        f"traction_{condition_id}:"
                                        f"{traction_metrics[condition_id].reject_reason}"
                                    )
                            except Exception as exc:
                                managed_errors[
                                    f"traction_{condition_id}_evaluator"
                                ] = str(exc)
                                reject_reasons.append(
                                    f"managed.traction_{condition_id}_evaluator_failed"
                                )
            except OSError as exc:
                managed_errors[f"traction_{condition_id}_launch"] = str(exc)
                task_last_error = str(exc)
                reject_reasons.append(
                    f"managed.traction_{condition_id}_launch_failed"
                )
            finally:
                if traction_session is not None:
                    traction_cleanup = _stop_session_and_check_ros_graph(
                        traction_session,
                        f"traction_{condition_id}",
                        managed_config,
                        shutdown_timeout_s,
                        ros_graph_checks,
                    )
                    if not _ros_graph_check_ok(traction_cleanup):
                        managed_runtime_blocked = True
                        managed_errors[
                            f"ros_graph_cleanup_traction_{condition_id}"
                        ] = (
                            _ros_graph_error_text(traction_cleanup)
                        )
                        reject_reasons.append("managed.ros_graph_cleanup_failed")
                task_metric = traction_metrics.get(condition_id)
                if task_metric is not None and not task_metric.valid:
                    task_status = "invalid"
                    task_last_error = task_metric.reject_reason
                elif len(reject_reasons) == task_reasons_start:
                    task_status = "complete"
                elif task_last_error is None:
                    task_last_error = "; ".join(
                        reject_reasons[task_reasons_start:]
                    )
                if task_callback is not None:
                    task_callback(
                        task_id,
                        "finished",
                        {
                            "status": task_status,
                            "output_dir": str(traction_output_base),
                            "raw_data_path": (
                                str(traction_run_dirs[condition_id])
                                if traction_run_dirs[condition_id] is not None
                                else None
                            ),
                            "last_error": task_last_error,
                        },
                    )
                if stop_on_task_failure and task_status == "failed":
                    managed_runtime_blocked = True
            if managed_runtime_blocked:
                break

    if (
        can_run_experiments
        and not managed_runtime_blocked
        and assets is not None
        and assets.mujoco_model_path is not None
        and frf_reuse_bundle is None
        and (
            collect_frf
            or any(task_id.startswith("frf_") for task_id in resume_task_outputs)
        )
    ):
        frf_output_base.mkdir(parents=True, exist_ok=True)
        frf_single_output_root = frf_output_base / "per_frequency"
        frf_analysis_input_dir = frf_single_output_root
        try:
            frequencies = _frf_frequencies(frf_config)
        except (TypeError, ValueError) as exc:
            frequencies = []
            managed_errors["frf_frequencies"] = str(exc)
            reject_reasons.append("managed.frf_invalid_frequencies")

        if not frequencies:
            reject_reasons.append("managed.frf_no_frequencies")

        for frequency_hz in frequencies:
            task_id = frf_task_id(frequency_hz)
            frequency_id = _frequency_run_id(frequency_hz)
            frequency_result: dict[str, Any] = {
                "frequency_hz": float(frequency_hz),
                "run_id": frequency_id,
                "output_base": str(frf_single_output_root / frequency_id),
                "source_experiment_dir": None,
                "source_run_dir": None,
                "analysis_run_dir": None,
            }
            frf_frequency_results.append(frequency_result)
            frf_session: SimulationSession | None = None
            resumed_run = resume_task_outputs.get(task_id)
            if resumed_run is not None:
                frequency_result.update(
                    {
                        "source_run_dir": str(resumed_run),
                        "analysis_run_dir": str(resumed_run),
                        "resumed": True,
                    }
                )
                continue

            if not collect_frf:
                continue

            if task_callback is not None:
                task_callback(task_id, "started", {})
            task_reasons_start = len(reject_reasons)
            task_status = "failed"
            task_last_error: str | None = None

            preflight_check = _wait_for_clean_ros_graph(
                f"frf.{frequency_id}.preflight",
                managed_config,
                "ros_graph_preflight_wait_s",
            )
            ros_graph_checks.append(preflight_check)
            frequency_result["preflight_ros_graph"] = preflight_check
            if not _ros_graph_check_ok(preflight_check):
                managed_runtime_blocked = True
                task_last_error = _ros_graph_error_text(preflight_check)
                managed_errors[f"ros_graph_preflight_{frequency_id}"] = task_last_error
                reject_reasons.append("managed.ros_graph_preflight_failed")
                if task_callback is not None:
                    task_callback(
                        task_id,
                        "finished",
                        {
                            "status": "failed",
                            "output_dir": frequency_result["output_base"],
                            "raw_data_path": None,
                            "last_error": task_last_error,
                        },
                    )
                break

            try:
                frf_launch_command = _launch_command_for_assets(
                    assets,
                    assets.mujoco_model_path,
                    default_traction_config,
                    traction_tendon_enabled=False,
                )
                frf_session = SimulationSession(
                    frf_launch_command,
                    logs_dir / "frf_mujoco" / frequency_id,
                )
                frf_sessions.append(frf_session)
                frf_session.start()
                time.sleep(startup_wait_s)
                if not frf_session.running():
                    reject_reasons.append("managed.frf_launch_exited")
                    continue

                frf_controller_check = _wait_for_active_controllers(
                    f"frf.{frequency_id}.controllers_ready",
                    managed_config,
                )
                controller_checks.append(frf_controller_check)
                frequency_result["controller_readiness"] = frf_controller_check
                if not _controller_check_ok(frf_controller_check):
                    managed_runtime_blocked = True
                    managed_errors[f"controller_readiness_{frequency_id}"] = (
                        _controller_error_text(frf_controller_check)
                    )
                    reject_reasons.append("managed.controller_readiness_failed")
                    continue

                frequency_output_base = frf_single_output_root / frequency_id
                frequency_output_base.mkdir(parents=True, exist_ok=True)
                single_frf_config = _single_frequency_frf_config(frf_config, frequency_hz)
                frf_command = build_frf_command_from_config(
                    single_frf_config,
                    frequency_output_base,
                )
                frf_run_ref = run_command_ref(
                    f"frf_recorder_{frequency_id}",
                    frf_command,
                    frequency_output_base,
                    logs_dir / "frf_recorders",
                    execute=True,
                    timeout_s=frf_timeout_from_config(
                        single_frf_config,
                        command_timeout_margin_s,
                    ),
                )
                frf_run_refs.append(frf_run_ref)
                if frf_run_ref.timed_out:
                    reject_reasons.append("managed.frf_recorder_timeout")
                elif frf_run_ref.returncode not in (0, None):
                    reject_reasons.append("managed.frf_recorder_failed")

                frf_experiment_dir = latest_frequency_response_experiment(frequency_output_base)
                frf_run_ref.discovered_output = frf_experiment_dir
                frequency_result["recorder"] = frf_run_ref.to_dict()
                frequency_result["source_experiment_dir"] = (
                    str(frf_experiment_dir) if frf_experiment_dir else None
                )
                if frf_experiment_dir is None:
                    reject_reasons.append("managed.frf_output_missing")
                    continue

                source_run_dir = _single_frequency_run_dir(
                    frf_experiment_dir,
                    frequency_hz,
                )
                if source_run_dir is None:
                    reject_reasons.append("managed.frf_run_missing")
                    continue
                frequency_result["source_run_dir"] = str(source_run_dir)
                frequency_result["analysis_run_dir"] = str(source_run_dir)
            except OSError as exc:
                managed_errors[f"frf_launch_{frequency_id}"] = str(exc)
                task_last_error = str(exc)
                reject_reasons.append("managed.frf_launch_failed")
            except Exception as exc:
                managed_errors[f"frf_collect_{frequency_id}"] = str(exc)
                task_last_error = str(exc)
                reject_reasons.append("managed.frf_collect_failed")
            finally:
                if frf_session is not None:
                    frf_cleanup = _stop_session_and_check_ros_graph(
                        frf_session,
                        f"frf.{frequency_id}",
                        managed_config,
                        shutdown_timeout_s,
                        ros_graph_checks,
                    )
                    frequency_result["cleanup_ros_graph"] = frf_cleanup
                    if not _ros_graph_check_ok(frf_cleanup):
                        managed_runtime_blocked = True
                        managed_errors[f"ros_graph_cleanup_{frequency_id}"] = (
                            _ros_graph_error_text(frf_cleanup)
                        )
                        reject_reasons.append("managed.ros_graph_cleanup_failed")
                    frequency_result["session"] = frf_session.to_dict()
                if len(reject_reasons) == task_reasons_start:
                    task_status = "complete"
                elif task_last_error is None:
                    task_last_error = "; ".join(
                        reject_reasons[task_reasons_start:]
                    )
                if task_callback is not None:
                    task_callback(
                        task_id,
                        "finished",
                        {
                            "status": task_status,
                            "output_dir": frequency_result["output_base"],
                            "raw_data_path": frequency_result.get("source_run_dir"),
                            "last_error": task_last_error,
                        },
                    )
                if stop_on_task_failure and task_status == "failed":
                    managed_runtime_blocked = True
            if managed_runtime_blocked:
                break

        collected_runs = [
            result
            for result in frf_frequency_results
            if result.get("source_run_dir") is not None
        ]
        if collected_runs and frf_analysis_input_dir is not None:
            try:
                frf_per_frequency_metadata_path = _write_per_frequency_frf_metadata(
                    frf_analysis_input_dir,
                    frf_config,
                    frf_frequency_results,
                )
                frf_analysis_result = _run_frf_analysis(
                    frf_analysis_input_dir,
                    frf_analysis_dir,
                    config.frf_analyzer,
                )
                frf_summary_path = Path(str(frf_analysis_result["summary_csv"]))
                if config.frf_real_reference is not None:
                    frf_metric = evaluate_frf_offline(
                        frf_summary_path,
                        config.frf_real_reference,
                    )
                    if not frf_metric.valid:
                        reject_reasons.append(f"frf:{frf_metric.reject_reason}")
            except (Exception, SystemExit) as exc:
                managed_errors["frf_analysis"] = str(exc)
                reject_reasons.append("managed.frf_analysis_failed")
        elif frequencies and collect_frf:
            reject_reasons.append("managed.frf_output_missing")

    if frf_reuse_bundle is None:
        frf_source_raw_data_paths = [
            str(result["source_run_dir"])
            for result in frf_frequency_results
            if result.get("source_run_dir")
        ]

    metadata_validation = validate_runtime_metadata_multi(
        parameters,
        traction_run_dirs=traction_run_dirs,
        frf_sim_summary=(
            None if frf_reuse_bundle is not None else frf_summary_path
        ),
    )
    reject_reasons.extend(metadata_validation.reject_reasons)
    reject_reasons = list(dict.fromkeys(reject_reasons))

    cost = compute_multispeed_total_cost(
        {
            condition_id: (
                traction_metrics[condition_id].J_tendon_point
                if condition_id in traction_metrics else None
            )
            for condition_id in TRACTION_CONDITION_IDS
        },
        frf_metric.J_frf if frf_metric else None,
        reject_reasons=reject_reasons,
    )

    provenance = _base_provenance("managed", config, assets, started)
    provenance["experiment_scope"] = experiment_scope
    provenance["metadata_paths"] = {
        "traction_run_metadata": {
            condition_id: (
                str(run_dir / "run_metadata.json") if run_dir else None
            )
            for condition_id, run_dir in traction_run_dirs.items()
        },
        "frf_sim_summary": str(frf_summary_path) if frf_summary_path else None,
        "frf_analysis_input": (
            str(frf_analysis_input_dir) if frf_analysis_input_dir else None
        ),
        "frf_per_frequency_metadata": (
            str(frf_per_frequency_metadata_path)
            if frf_per_frequency_metadata_path
            else None
        ),
        "frf_combined_metadata": None,
        "frf_real_reference": (
            str(config.frf_real_reference)
            if config.frf_real_reference else None
        ),
        "frf_source_raw_data_paths": list(frf_source_raw_data_paths),
    }
    provenance["runtime_metadata_validation"] = metadata_validation.to_dict()
    provenance["launch_commands"] = [
        session.command
        for session in list(traction_sessions.values()) + frf_sessions
        if session is not None
    ]
    analyzer_commands: list[list[str]] = []
    for condition_id in TRACTION_CONDITION_IDS:
        axis = condition_axis(condition_id)
        traction_run_dir = traction_run_dirs.get(condition_id)
        if traction_run_dir is None:
            continue
        traction_analyzer_config = config.traction_analyzer
        traction_analyzer_command = [
            "ros2",
            "run",
            "rbkairos",
            "analyze_traction_tendon.py",
            str(traction_run_dir),
            "--axis",
            axis,
            "--steady-window",
            str(_managed_float(traction_analyzer_config, "steady_window", 2.0)),
            "--steady-window-step",
            str(_managed_float(traction_analyzer_config, "steady_window_step", 0.25)),
            "--min-observation",
            str(_managed_float(traction_analyzer_config, "min_observation", 2.0)),
            "--required-stable-windows",
            str(int(traction_analyzer_config.get("required_stable_windows", 4))),
            "--max-wait",
            str(_managed_float(traction_analyzer_config, "max_wait", 60.0)),
            "--min-active-ratio",
            str(
                _managed_float(
                    traction_analyzer_config,
                    "min_tendon_active_ratio",
                    0.95,
                )
            ),
            "--max-force-cv",
            str(_managed_float(traction_analyzer_config, "max_force_cv", 0.05)),
            "--max-force-slope",
            str(_managed_float(traction_analyzer_config, "max_force_slope_n_s", 0.5)),
            "--max-constraint-extension",
            str(
                _managed_float(
                    traction_analyzer_config,
                    "max_constraint_extension_m",
                    0.005,
                )
            ),
            "--max-steady-base-velocity",
            str(
                _managed_float(
                    traction_analyzer_config,
                    "max_steady_base_velocity_mps",
                    0.005,
                )
            ),
            "--max-i-term-slope",
            str(
                _managed_float(
                    traction_analyzer_config,
                    "max_i_term_slope_nm_s",
                    0.005,
                )
            ),
            "--min-i-clamp-ratio",
            str(_managed_float(traction_analyzer_config, "min_i_clamp_ratio", 0.95)),
        ]
        analyzer_commands.append(traction_analyzer_command)
    if (
        frf_analysis_input_dir is not None
        and frf_analysis_input_dir.exists()
        and any(result.get("source_run_dir") for result in frf_frequency_results)
    ):
        frf_analyzer_config = config.frf_analyzer
        frf_analyzer_command = [
            "ros2",
            "run",
            "rbkairos",
            "analyze_frequency_response_fft.py",
            "--experiment-dir",
            str(frf_analysis_input_dir),
            "--output-dir",
            str(frf_analysis_dir),
            "--discard-cycles",
            str(
                _managed_float(
                    frf_analyzer_config,
                    "discard_cycles",
                    _managed_float(frf_analyzer_config, "default_discard_cycles", 2.0),
                )
            ),
            "--ramp-cycles",
            str(_managed_float(frf_analyzer_config, "ramp_cycles", 1.0)),
            "--analysis-cycles",
            str(_managed_int(frf_analyzer_config, "analysis_cycles", 4)),
            "--grid-rate-hz",
            str(
                _managed_float(
                    frf_analyzer_config,
                    "grid_rate_hz",
                    _managed_float(frf_analyzer_config, "resample_rate", 50.0),
                )
            ),
        ]
        if not _managed_bool(frf_analyzer_config, "plots", True):
            frf_analyzer_command.append("--skip-plots")
        if not _managed_bool(frf_analyzer_config, "html", True):
            frf_analyzer_command.append("--skip-html")
        analyzer_commands.append(frf_analyzer_command)
    provenance["analyzer_commands"] = analyzer_commands
    provenance["managed"] = {
        "startup_wait_s": startup_wait_s,
        "shutdown_timeout_s": shutdown_timeout_s,
        "force_kill_timeout_s": _managed_float(
            managed_config,
            "force_kill_timeout_s",
            1.0,
        ),
        "command_timeout_margin_s": command_timeout_margin_s,
        "ros_graph_checks": ros_graph_checks,
        "controller_checks": controller_checks,
        "controller_check_config": {
            "enabled": _controller_readiness_checks_enabled(managed_config),
            "controller_manager": str(
                managed_config.get("controller_manager_node", "/controller_manager")
            ),
            "required_active": _managed_string_list(
                managed_config,
                "controller_required_active",
                MANAGED_REQUIRED_ACTIVE_CONTROLLERS,
            ),
            "readiness_wait_s": _managed_float(
                managed_config,
                "controller_readiness_wait_s",
                3.0,
            ),
            "poll_s": _managed_float(
                managed_config,
                "controller_readiness_poll_s",
                0.25,
            ),
            "list_timeout_s": _managed_float(
                managed_config,
                "controller_list_timeout_s",
                2.0,
            ),
            "method": "rclpy_service_subprocess",
        },
        "ros_graph_check_config": {
            "enabled": _ros_graph_checks_enabled(managed_config),
            "blocking_nodes": _managed_string_list(
                managed_config,
                "ros_graph_blocking_nodes",
                MANAGED_BLOCKING_ROS_NODES,
            ),
            "preflight_wait_s": _managed_float(
                managed_config,
                "ros_graph_preflight_wait_s",
                0.5,
            ),
            "cleanup_wait_s": _managed_float(
                managed_config,
                "ros_graph_cleanup_wait_s",
                5.0,
            ),
            "force_kill_wait_s": _managed_float(
                managed_config,
                "ros_graph_force_kill_wait_s",
                1.0,
            ),
            "use_daemon": _managed_bool(managed_config, "ros_graph_use_daemon", False),
            "spin_time_s": _managed_float(managed_config, "ros_graph_spin_time_s", 0.1),
            "log_dir": str(
                managed_config.get("ros_graph_log_dir", "/tmp/rbkairos_ros_graph_checks")
            ),
            "poll_s": _managed_float(managed_config, "ros_graph_poll_s", 0.25),
            "node_list_timeout_s": _managed_float(
                managed_config,
                "ros_graph_node_list_timeout_s",
                1.0,
            ),
        },
        "traction_sessions": {
            axis: session.to_dict() for axis, session in traction_sessions.items()
        },
        "frf_session": frf_sessions[0].to_dict() if len(frf_sessions) == 1 else None,
        "frf_sessions": [session.to_dict() for session in frf_sessions],
        "traction_runs": {
            axis: run_ref.to_dict() for axis, run_ref in traction_run_refs.items()
        },
        "traction_output_dirs": {
            axis: str(path) for axis, path in traction_output_bases.items()
        },
        "frf_run": frf_run_refs[0].to_dict() if len(frf_run_refs) == 1 else None,
        "frf_runs": [run_ref.to_dict() for run_ref in frf_run_refs],
        "frf_per_frequency_root": (
            str(frf_single_output_root) if frf_single_output_root else None
        ),
        "frf_analysis_input": (
            str(frf_analysis_input_dir) if frf_analysis_input_dir else None
        ),
        "frf_per_frequency_metadata": (
            str(frf_per_frequency_metadata_path)
            if frf_per_frequency_metadata_path
            else None
        ),
        "frf_combined_experiment": None,
        "frf_frequency_results": frf_frequency_results,
        "frf_reused_from": frf_reused_from,
        "frf_nonbinding_certificate_fingerprint": (
            frf_nonbinding_certificate.get("certificate_fingerprint")
            if frf_nonbinding_certificate
            else None
        ),
        "frf_analysis": (
            {
                "summary_csv": frf_analysis_result.get("summary_csv"),
                "output_dir": frf_analysis_result.get("output_dir"),
                "run_count": frf_analysis_result.get("run_count"),
                "valid_run_count": frf_analysis_result.get("valid_run_count"),
            }
            if frf_analysis_result
            else None
        ),
        "errors": managed_errors,
    }
    provenance["wall_time"]["duration_s"] = time.time() - started

    if persist_result:
        return _write_result(
            parameters=parameters,
            cost=cost,
            traction=traction_metrics,
            frf=frf_metric,
            provenance=provenance,
            output_dir=output_dir,
            frf_reused_from=frf_reused_from,
            frf_nonbinding_certificate=frf_nonbinding_certificate,
            frf_source_raw_data_paths=frf_source_raw_data_paths,
        )
    return EvaluationResult(
        parameters=parameters,
        cost=cost,
        traction=traction_metrics,
        frf=frf_metric,
        provenance=provenance,
        output_dir=output_dir,
        frf_reused_from=frf_reused_from,
        frf_nonbinding_certificate=frf_nonbinding_certificate,
        frf_source_raw_data_paths=frf_source_raw_data_paths,
    )


def run_managed_campaign_candidate(
    *,
    candidate: dict[str, Any],
    config: EvaluationConfig,
    output_dir: Path,
    completed_task_outputs: dict[str, str],
    frf_reuse_bundle: dict[str, Any] | None,
    task_callback: Callable[[str, str, dict[str, Any]], None],
    strict: bool,
    experiment_scope: str = "all",
):
    """Run pending physical tasks for one campaign candidate."""
    from evaluation.campaign_executor import CampaignCandidateExecution

    parameters = EvaluationParameters(**candidate["parameters"])
    observed_outputs: dict[str, str] = {}

    def observe_task(
        task_id: str,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        task_callback(task_id, event, payload)
        if (
            event == "finished"
            and payload.get("status") in {"complete", "invalid"}
            and payload.get("raw_data_path")
        ):
            observed_outputs[task_id] = str(payload["raw_data_path"])

    result = evaluate_managed(
        parameters,
        config,
        output_dir=output_dir,
        create_assets=True,
        strict=strict,
        resume_task_outputs=completed_task_outputs,
        frf_reuse_bundle=frf_reuse_bundle,
        task_callback=observe_task,
        stop_on_task_failure=True,
        persist_result=False,
        experiment_scope=experiment_scope,
    )

    if frf_reuse_bundle is not None:
        return CampaignCandidateExecution(
            result=result,
            frf_frequency_evidence=[],
            frf_source_raw_data_paths=dict(
                frf_reuse_bundle["source_raw_data_paths"]
            ),
        )

    frf_raw_paths = {
        task_id: str(path)
        for task_id, path in {
            **completed_task_outputs,
            **observed_outputs,
        }.items()
        if task_id.startswith("frf_")
    }
    evidence: list[dict[str, Any]] = []
    for task_id in candidate["tasks"]["frf"]:
        raw_path = frf_raw_paths.get(task_id)
        if raw_path is None:
            continue
        metadata_path = Path(raw_path) / "run_metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        raw_evidence = metadata.get("frf_pid_nonbinding_evidence")
        if isinstance(raw_evidence, dict):
            evidence.append(raw_evidence)
    return CampaignCandidateExecution(
        result=result,
        frf_frequency_evidence=evidence,
        frf_source_raw_data_paths=frf_raw_paths,
    )
