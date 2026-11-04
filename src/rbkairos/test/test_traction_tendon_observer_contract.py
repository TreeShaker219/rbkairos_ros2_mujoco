from pathlib import Path
import re
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
HEADER = PACKAGE_ROOT / "include" / "rbkairos" / "traction_tendon_observer_plugin.hpp"
SOURCE = PACKAGE_ROOT / "src" / "traction_tendon_observer_plugin.cpp"
PLUGIN_XML = PACKAGE_ROOT / "rbkairos_mujoco_plugins.xml"

EXPECTED_FIELDS = (
    "sim_time_s",
    "tendon_length_m",
    "tendon_length0_m",
    "lower_limit_m",
    "upper_limit_m",
    "extension_over_upper_limit_m",
    "tension_n",
    "limit_active_flag",
    "force_x_n",
    "force_y_n",
    "force_z_n",
    "attach_x_m",
    "attach_y_m",
    "attach_z_m",
    "anchor_x_m",
    "anchor_y_m",
    "anchor_z_m",
)


def read(path):
    assert path.is_file(), f"missing observer file: {path}"
    return path.read_text(encoding="utf-8")


def test_observer_is_registered_as_a_physics_plugin():
    root = ET.parse(PLUGIN_XML).getroot()
    classes = {
        element.get("type"): element
        for element in root.findall("./library/class")
    }
    assert set(classes) == {
        "rbkairos::BaseGroundTruthPlugin",
        "rbkairos::TractionTendonObserverPlugin",
    }
    observer = classes["rbkairos::TractionTendonObserverPlugin"]
    assert observer.get("base_class_type") == "mujoco_sim_ros2::MujocoPhysicsPlugin"
    libraries = {
        child.get("path")
        for child in root.findall("./library")
        if child.find("./class[@type='rbkairos::TractionTendonObserverPlugin']") is not None
    }
    assert libraries == {"rbkairos_traction_tendon_observer_plugin"}


def test_observer_source_cannot_apply_physics_forces():
    source = read(SOURCE)
    forbidden = (
        "mj_applyFT",
        "qfrc_applied",
        "xfrc_applied",
        "qfrc_actuator",
        "mjcb_control",
    )
    for token in forbidden:
        assert token not in source


def test_observer_uses_native_tendon_state_and_limit_force_sensor():
    source = read(SOURCE)
    required = (
        "ten_length",
        "tendon_length0",
        "tendon_range",
        "tendon_efcadr",
        "sensordata",
        "sensor_adr",
        "sensor_needstage",
        "sensor_datatype",
        "sensor_noise",
        "sensor_cutoff",
        "site_xpos",
        "tendon_adr",
        "tendon_num",
        "tendon_stiffness",
        "tendon_damping",
        "tendon_frictionloss",
        "actuator_trntype",
        "actuator_trnid",
        "efc_type",
        "efc_id",
        "wrap_type",
        "wrap_objid",
    )
    for token in required:
        assert token in source
    assert "std::fabs" not in source


def test_observer_accepts_mujoco_compiled_tendon_limit_sensor_datatype():
    source = read(SOURCE)

    # MuJoCo 3.3.x compiles <tendonlimitfrc> as a scalar REAL sensor even
    # though a valid upper-limit force is non-negative in this experiment.
    assert "mjDATATYPE_REAL" in source
    assert "mjDATATYPE_POSITIVE" not in source


def test_observer_resets_publish_schedule_when_the_model_changes():
    header = read(HEADER)
    source = read(SOURCE)

    assert "void reset_publish_schedule()" in header
    assert "model != model_" in source
    assert source.count("reset_publish_schedule();") >= 2


def test_observer_publishes_the_version_one_17_field_schema():
    header = read(HEADER)
    source = read(SOURCE)

    assert re.search(r"kStateSchemaVersion\s*=\s*1", header)
    assert re.search(r"kStateFieldCount\s*=\s*17", header)
    assert "std_msgs::msg::Float64MultiArray" in header
    for index, field_name in enumerate(EXPECTED_FIELDS):
        assert f'"{field_name}"' in source, f"field {index} ({field_name}) is undocumented"
    assert '"/traction_tendon/state"' in source
    assert '"/traction_tendon/wrench"' in source
