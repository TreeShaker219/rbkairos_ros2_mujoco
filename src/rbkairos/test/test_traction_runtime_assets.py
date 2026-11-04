import xml.etree.ElementTree as ET
from pathlib import Path
import sys

import mujoco
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from evaluation.parameters import EvaluationParameters  # noqa: E402
from evaluation.runtime_assets import create_runtime_assets  # noqa: E402


def test_runtime_assets_generate_two_tendon_scenes_and_plain_frf_base(tmp_path):
    source_model = PACKAGE_ROOT / "mujoco" / "rbkairos.xml"
    assets = create_runtime_assets(
        EvaluationParameters(kp=51.0, ki=0.2, i_max=3.0),
        tmp_path,
        source_urdf=PACKAGE_ROOT / "urdf" / "rbkairos.urdf",
        source_mjcf=source_model,
        source_traction_scenes={
            axis: PACKAGE_ROOT / "mujoco" / f"rbkairos_scene_tendon_test_{axis}.xml"
            for axis in ("x", "y")
        },
        source_config=PACKAGE_ROOT / "config" / "mujoco_mecanum_drive.yaml",
    )

    assert set(assets.traction_mujoco_model_paths) == {"x", "y"}
    base = mujoco.MjModel.from_xml_path(str(assets.mujoco_model_path))
    assert base.ntendon == 0
    for axis, scene_path in assets.traction_mujoco_model_paths.items():
        assert scene_path.parent == assets.mujoco_model_path.parent
        include = ET.parse(scene_path).getroot().find("./include")
        assert include is not None
        assert include.get("file") == assets.mujoco_model_path.name
        scene = mujoco.MjModel.from_xml_path(str(scene_path))
        assert scene.ntendon == 1
        assert scene.nsensor == 1
        assert f"tendon_test_{axis}" in scene_path.name

    compiled_pair_mu = {float(row[1]) for row in base.pair_friction}
    assert len(compiled_pair_mu) == 1
    assert next(iter(compiled_pair_mu)) == pytest.approx(0.7)
    source_pair_mu = {
        float(pair.get("friction").split()[1])
        for pair in ET.parse(source_model).getroot().findall(".//contact/pair")
    }
    assert source_pair_mu == {0.7}
    generated_urdf = ET.parse(assets.urdf_path).getroot()
    wheel_control = {
        joint.get("name"): {
            param.get("name"): float(param.text)
            for param in joint.findall("param")
        }
        for joint in generated_urdf.findall(".//ros2_control/joint")
        if "wheel_joint" in str(joint.get("name"))
    }
    assert len(wheel_control) == 4
    for values in wheel_control.values():
        assert values["velocity_kp"] == pytest.approx(51.0)
        assert values["velocity_ki"] == pytest.approx(0.2)
        assert values["velocity_i_max"] == pytest.approx(3.0)
        assert values["velocity_i_min"] == pytest.approx(-3.0)
        assert values["velocity_kd"] == pytest.approx(0.0)
    wheel_effort = {
        float(joint.find("limit").get("effort"))
        for joint in generated_urdf.findall(".//joint")
        if joint.get("name") in wheel_control and joint.find("limit") is not None
    }
    assert wheel_effort == {100.0}
    assert "spring" not in str(assets.to_dict()).lower()
