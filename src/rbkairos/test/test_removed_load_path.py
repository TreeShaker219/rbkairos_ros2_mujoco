from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DELETED_PATHS = (
    PACKAGE_ROOT / "include" / "rbkairos" / "virtual_spring_plugin.hpp",
    PACKAGE_ROOT / "src" / "virtual_spring_plugin.cpp",
    PACKAGE_ROOT / "scripts" / "virtual_spring_pull_experiment.py",
    PACKAGE_ROOT / "scripts" / "analyze_virtual_spring_pull.py",
    PACKAGE_ROOT / "mujoco" / "rbkairos_scene_spring_test.xml",
)
PRODUCTION_TEXT_SUFFIXES = {".cpp", ".hpp", ".py", ".xml", ".yaml"}
REMOVED_IDENTIFIERS = (
    "virtualspring",
    "virtual_spring",
    "virtual spring",
    "virtual-spring",
    "spring_attach",
    "spring_wall",
    "scene_spring",
    "spring_experiments",
    "j_spring",
    "f_spring",
)


def production_text_files():
    yield PACKAGE_ROOT / "CMakeLists.txt"
    for path in PACKAGE_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in PRODUCTION_TEXT_SUFFIXES:
            continue
        if "test" in path.relative_to(PACKAGE_ROOT).parts:
            continue
        yield path


def test_removed_load_files_and_production_identifiers_are_absent():
    assert not [path for path in DELETED_PATHS if path.exists()]

    residuals = []
    for path in production_text_files():
        text = path.read_text(encoding="utf-8").lower()
        for identifier in REMOVED_IDENTIFIERS:
            if identifier in text:
                residuals.append(f"{path.relative_to(PACKAGE_ROOT)}:{identifier}")
    assert residuals == []


def test_install_rules_exclude_generated_python_caches():
    cmake = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert cmake.count('PATTERN "__pycache__" EXCLUDE') >= 2


def test_plugin_registry_and_base_config_only_keep_production_plugins():
    plugin_root = ET.parse(PACKAGE_ROOT / "rbkairos_mujoco_plugins.xml").getroot()
    plugin_classes = {
        node.get("type") for node in plugin_root.findall("./library/class")
    }
    assert plugin_classes == {
        "rbkairos::BaseGroundTruthPlugin",
        "rbkairos::TractionTendonObserverPlugin",
    }

    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "mujoco_mecanum_drive.yaml").read_text(
            encoding="utf-8"
        )
    )
    parameters = config["mujoco_sim_ros2_node"]["ros__parameters"]
    assert parameters["physics_plugins"] == [
        "mujoco_ros2_control::MujocoRos2ControlPlugin",
        "rbkairos::BaseGroundTruthPlugin",
    ]
    assert "traction_tendon" not in parameters


def test_tendon_sites_and_protected_physical_parameters_remain():
    model_root = ET.parse(PACKAGE_ROOT / "mujoco" / "rbkairos.xml").getroot()
    sites = {site.get("name") for site in model_root.findall(".//site")}
    assert {"traction_attach_x", "traction_attach_y"} <= sites
    assert not {"spring_attach", "spring_attach_y"} & sites

    contact_friction = {
        pair.get("name"): pair.get("friction")
        for pair in model_root.findall(".//contact/pair")
    }
    assert all(
        contact_friction[name] == "0.01 0.7"
        for name in ("fric_lf", "fric_rf", "fric_lb", "fric_rb")
    )

    urdf_root = ET.parse(PACKAGE_ROOT / "urdf" / "rbkairos.urdf").getroot()
    wheel_joints = [
        joint
        for joint in urdf_root.findall(".//ros2_control/joint")
        if "wheel_joint" in str(joint.get("name"))
    ]
    assert len(wheel_joints) == 4
    for joint in wheel_joints:
        parameters = {
            param.get("name"): param.text for param in joint.findall("param")
        }
        assert parameters["velocity_i_min"] == "-1.0"
        assert parameters["velocity_i_max"] == "1.0"
