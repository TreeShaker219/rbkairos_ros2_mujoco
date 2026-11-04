from copy import deepcopy
import importlib.util
import os
from pathlib import Path

import yaml
import pytest


os.environ.setdefault("ROS_LOG_DIR", "/tmp/rbkairos_ros_log")


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE_ROOT / "config" / "mujoco_mecanum_drive.yaml"
LAUNCH = PACKAGE_ROOT / "launch" / "mujoco_mecanum_drive.launch.py"
OBSERVER_CLASS = "rbkairos::TractionTendonObserverPlugin"


def load_launch_module():
    spec = importlib.util.spec_from_file_location("rbkairos_mujoco_launch", LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ros_parameters():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return config["mujoco_sim_ros2_node"]["ros__parameters"]


def observer_options():
    return {
        "tendon_name": "traction_cable_y",
        "sensor_name": "traction_cable_y_limit_force",
        "attach_site_name": "traction_attach_y",
        "anchor_site_name": "traction_anchor_y",
        "publish_rate_hz": 120.0,
        "state_topic": "/test/traction_state",
        "wrench_topic": "/test/traction_wrench",
        "frame_id": "world",
    }


def test_base_yaml_does_not_load_or_configure_the_observer():
    parameters = ros_parameters()

    assert parameters["physics_plugins"] == [
        "mujoco_ros2_control::MujocoRos2ControlPlugin",
        "rbkairos::BaseGroundTruthPlugin",
    ]
    assert "traction_tendon" not in parameters


def test_disabled_launch_configuration_keeps_the_observer_absent():
    module = load_launch_module()
    parameters = deepcopy(ros_parameters())
    original_plugins = list(parameters["physics_plugins"])
    parameters["physics_plugins"].insert(1, OBSERVER_CLASS)
    parameters["traction_tendon"] = {"stale": True}

    module.configure_traction_tendon(parameters, enabled=False, options=observer_options())

    assert OBSERVER_CLASS not in parameters["physics_plugins"]
    assert "traction_tendon" not in parameters
    assert parameters["physics_plugins"] == original_plugins


def test_enabled_launch_configuration_injects_one_observer_and_its_parameters():
    module = load_launch_module()
    parameters = deepcopy(ros_parameters())
    options = observer_options()

    module.configure_traction_tendon(parameters, enabled=True, options=options)
    module.configure_traction_tendon(parameters, enabled=True, options=options)

    assert parameters["physics_plugins"].count(OBSERVER_CLASS) == 1
    assert parameters["traction_tendon"] == options


def test_invalid_physics_plugin_configuration_fails_fast():
    module = load_launch_module()
    parameters = {"physics_plugins": "not-a-list"}

    with pytest.raises(ValueError, match="physics_plugins must be a list"):
        module.configure_traction_tendon(parameters, enabled=True, options=observer_options())


def test_launch_declares_tendon_observer_arguments_disabled_by_default():
    module = load_launch_module()
    description = module.generate_launch_description()
    arguments = {
        entity.name: entity
        for entity in description.entities
        if entity.__class__.__name__ == "DeclareLaunchArgument"
    }

    expected_names = {
        "traction_tendon_enabled",
        "traction_tendon_name",
        "traction_tendon_sensor_name",
        "traction_tendon_attach_site_name",
        "traction_tendon_anchor_site_name",
        "traction_tendon_publish_rate_hz",
        "traction_tendon_state_topic",
        "traction_tendon_wrench_topic",
        "traction_tendon_frame_id",
    }
    assert expected_names <= set(arguments)
    assert arguments["traction_tendon_enabled"].default_value[0].text == "false"


def test_initial_arm_hold_preserves_exact_urdf_targets():
    module = load_launch_module()
    positions = module.initial_arm_positions((PACKAGE_ROOT / "urdf/rbkairos.urdf").read_text())
    assert positions == [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
    xml = "<robot><ros2_control>" + "".join(
        f'<joint name="{name}"><state_interface name="position">'
        f'<param name="initial_value">{i / 10}</param></state_interface></joint>'
        for i, name in enumerate(module.ARM_HOLD_JOINTS)
    ) + "</ros2_control></robot>"
    assert module.initial_arm_positions(xml) == [i / 10 for i in range(6)]
    with pytest.raises(ValueError, match="Non-finite"):
        module.initial_arm_positions(xml.replace('>0.0<', '>nan<'))
    with pytest.raises(ValueError, match="Expected one"):
        module.initial_arm_positions('<robot/>')


def test_initial_arm_hold_enabled_by_default():
    module = load_launch_module()
    args = {x.name: x for x in module.generate_launch_description().entities
            if x.__class__.__name__ == "DeclareLaunchArgument"}
    assert args["initialize_arm_hold"].default_value[0].text == "true"
