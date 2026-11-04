import json
from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from evaluation import runners  # noqa: E402


def test_traction_command_and_timeout_use_adaptive_maximum_budget(tmp_path):
    config = {
        "axis": "y",
        "velocity_y": 0.1,
        "warmup_duration": 1.0,
        "ramp_duration": 0.5,
        "min_observation_duration": 2.0,
        "max_wait_duration": 60.0,
        "post_stop_duration": 0.5,
        "steady_window": 2.0,
        "steady_window_step": 0.25,
        "required_stable_windows": 4,
        "max_force_slope": 0.5,
        "max_i_term_slope": 0.005,
        "min_i_clamp_ratio": 0.95,
        "publish_rate": 100.0,
        "observer_publish_rate": 100.0,
        "startup_timeout": 10.0,
        "model_file": "/tmp/tendon_y.xml",
        "tendon_name": "traction_cable_y",
        "sensor_name": "traction_cable_y_limit_force",
        "attach_site_name": "traction_attach_y",
        "anchor_site_name": "traction_anchor_y",
        "expected_ur5_preset": "up",
    }

    command = runners.build_traction_command_from_config(config, tmp_path)

    joined = " ".join(command)
    assert command[:4] == ["ros2", "run", "rbkairos", "traction_tendon_experiment.py"]
    assert "--constant-duration" not in joined
    assert "--min-observation-duration 2.0" in joined
    assert "--max-wait-duration 60.0" in joined
    assert "--required-stable-windows 4" in joined
    assert "--max-i-term-slope 0.005" in joined
    assert "--tendon-name traction_cable_y" in joined
    assert "--model-file /tmp/tendon_y.xml" in joined
    assert "--expected-ur5-preset up" in joined
    assert "spring" not in joined.lower()
    assert "stall" not in joined.lower()
    assert runners.traction_timeout_from_config(config, 30.0) == 102.0
    assert not hasattr(runners, "build_spring_command_from_config")


def test_latest_traction_run_rejects_legacy_spring_directory(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "run_metadata.json").write_text(
        json.dumps(
            {
                "experiment_type": "virtual_spring_pull",
                "measurement_method": "virtual_spring",
            }
        ),
        encoding="utf-8",
    )
    current = tmp_path / "current"
    current.mkdir()
    (current / "run_metadata.json").write_text(
        json.dumps(
            {
                "experiment_type": "traction_tendon_pull",
                "measurement_method": "tendon_limit",
                "schema_version": 1,
            }
        ),
        encoding="utf-8",
    )
    (current / "traction_tendon_state.csv").write_text("header\n", encoding="utf-8")

    assert runners.latest_traction_tendon_run(tmp_path) == current
