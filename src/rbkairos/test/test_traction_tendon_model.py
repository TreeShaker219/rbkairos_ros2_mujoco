from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PACKAGE_ROOT / "mujoco"
BASE_MODEL = MODEL_DIR / "rbkairos.xml"

AXES = {
    "x": {
        "scene": MODEL_DIR / "rbkairos_scene_tendon_test_x.xml",
        "tendon": "traction_cable_x",
        "sensor": "traction_cable_x_limit_force",
        "attach": "traction_attach_x",
        "anchor": "traction_anchor_x",
        "attach_local": np.array([-0.32, 0.0, 0.09]),
        "attach_world": np.array([-0.32, 0.0, 0.217]),
        "anchor_world": np.array([-2.0, 0.0, 0.217]),
        "upper_limit": 1.681,
        "cross_axes": (1, 2),
        "axis_index": 0,
    },
    "y": {
        "scene": MODEL_DIR / "rbkairos_scene_tendon_test_y.xml",
        "tendon": "traction_cable_y",
        "sensor": "traction_cable_y_limit_force",
        "attach": "traction_attach_y",
        "anchor": "traction_anchor_y",
        "attach_local": np.array([0.0, -0.11, 0.09]),
        "attach_world": np.array([0.0, -0.11, 0.217]),
        "anchor_world": np.array([0.0, -2.0, 0.217]),
        "upper_limit": 1.891,
        "cross_axes": (0, 2),
        "axis_index": 1,
    },
}

SENSITIVITY_SCENES = (
    ("x", "iterations100", 100, np.array([0.02, 1.0])),
    ("y", "iterations100", 100, np.array([0.02, 1.0])),
    ("x", "hard_limit", 50, np.array([0.01, 1.0])),
    ("y", "hard_limit", 50, np.array([0.01, 1.0])),
)
WHEEL_JOINTS = (
    "left_front_wheel_joint",
    "right_front_wheel_joint",
    "left_back_wheel_joint",
    "right_back_wheel_joint",
)
UR5_JOINTS = (
    "ur5/shoulder_pan_joint",
    "ur5/shoulder_lift_joint",
    "ur5/elbow_joint",
    "ur5/wrist_1_joint",
    "ur5/wrist_2_joint",
    "ur5/wrist_3_joint",
)


def object_id(model, object_type, name):
    object_id_value = mujoco.mj_name2id(model, object_type, name)
    assert object_id_value >= 0, f"missing MuJoCo object {name!r}"
    return object_id_value


def load_model(path):
    assert path.is_file(), f"missing model: {path}"
    return mujoco.MjModel.from_xml_path(str(path))


def simulate_isolated_contact_force(scene, axis):
    """Remove ROS/PID timing and expose contact/tendon force stability directly."""
    model = load_model(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    wheel_dofs = [
        model.jnt_dofadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in WHEEL_JOINTS
    ]
    arm_dofs = [
        model.jnt_dofadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in UR5_JOINTS
    ]
    sensor_id = object_id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, AXES[axis]["sensor"]
    )
    sensor_address = model.sensor_adr[sensor_id]
    signs = np.array([1.0, 1.0, 1.0, 1.0])
    if axis == "y":
        signs = np.array([-1.0, 1.0, 1.0, -1.0])
    samples = []
    while data.time < 4.0:
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[arm_dofs] = data.qfrc_bias[arm_dofs]
        elapsed = data.time - 1.0
        if elapsed <= 0.0:
            scale = 0.0
        elif elapsed < 0.5:
            scale = 0.5 * (1.0 - np.cos(np.pi * elapsed / 0.5))
        else:
            scale = 1.0
        data.qfrc_applied[wheel_dofs] = 38.0 * scale * signs
        mujoco.mj_step(model, data)
        if data.time >= 3.0:
            samples.append(float(data.sensordata[sensor_address]))
    values = np.asarray(samples)
    return float(np.median(values)), float(np.std(values) / np.mean(values))


def test_base_model_compiles_without_a_tendon_constraint():
    model = load_model(BASE_MODEL)

    assert model.ntendon == 0
    assert model.nsensor == 0
    assert model.opt.cone == int(mujoco.mjtCone.mjCONE_PYRAMIDAL)


def test_base_model_only_keeps_face_center_traction_sites():
    model = load_model(BASE_MODEL)

    for legacy_name in ("spring_attach", "spring_attach_y"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, legacy_name) == -1

    for expected in AXES.values():
        site_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, expected["attach"])
        np.testing.assert_allclose(
            model.site_pos[site_id], expected["attach_local"], atol=1e-12
        )


@pytest.mark.parametrize("axis", ("x", "y"))
def test_tendon_scene_compiles_with_one_limit_only_cable(axis):
    expected = AXES[axis]
    model = load_model(expected["scene"])

    assert model.ntendon == 1
    assert model.opt.cone == int(mujoco.mjtCone.mjCONE_ELLIPTIC)
    np.testing.assert_allclose(
        model.pair_solreffriction,
        np.tile([-1.0, -1.0], (model.npair, 1)),
    )
    tendon_id = object_id(model, mujoco.mjtObj.mjOBJ_TENDON, expected["tendon"])
    assert bool(model.tendon_limited[tendon_id])
    np.testing.assert_allclose(
        model.tendon_range[tendon_id], [0.0, expected["upper_limit"]], atol=1e-12
    )
    assert model.tendon_stiffness[tendon_id] == pytest.approx(0.0)
    assert model.tendon_damping[tendon_id] == pytest.approx(0.0)
    assert model.tendon_frictionloss[tendon_id] == pytest.approx(0.0)

    tendon_actuators = np.flatnonzero(
        model.actuator_trntype == int(mujoco.mjtTrn.mjTRN_TENDON)
    )
    assert tendon_actuators.size == 0

    limit_force_sensor_ids = np.flatnonzero(
        model.sensor_type == int(mujoco.mjtSensor.mjSENS_TENDONLIMITFRC)
    )
    assert limit_force_sensor_ids.size == 1
    sensor_id = object_id(model, mujoco.mjtObj.mjOBJ_SENSOR, expected["sensor"])
    assert sensor_id == int(limit_force_sensor_ids[0])
    assert model.sensor_objtype[sensor_id] == int(mujoco.mjtObj.mjOBJ_TENDON)
    assert model.sensor_objid[sensor_id] == tendon_id
    assert model.sensor_dim[sensor_id] == 1
    assert 0 <= model.sensor_adr[sensor_id] < model.nsensordata

    attach_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, expected["attach"])
    anchor_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, expected["anchor"])
    path_start = model.tendon_adr[tendon_id]
    assert model.tendon_num[tendon_id] == 2
    assert np.all(
        model.wrap_type[path_start:path_start + 2]
        == int(mujoco.mjtWrap.mjWRAP_SITE)
    )
    assert list(model.wrap_objid[path_start:path_start + 2]) == [anchor_id, attach_id]

    base_link_id = object_id(
        model, mujoco.mjtObj.mjOBJ_BODY, "summit_xl_2/base_link"
    )
    assert model.site_bodyid[anchor_id] == 0
    assert model.site_bodyid[attach_id] == base_link_id


@pytest.mark.parametrize("axis", ("x", "y"))
def test_tendon_initial_geometry_is_axis_aligned_and_has_one_mm_slack(axis):
    expected = AXES[axis]
    model = load_model(expected["scene"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    tendon_id = object_id(model, mujoco.mjtObj.mjOBJ_TENDON, expected["tendon"])
    attach_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, expected["attach"])
    anchor_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, expected["anchor"])
    sensor_id = object_id(model, mujoco.mjtObj.mjOBJ_SENSOR, expected["sensor"])

    attach_world = data.site_xpos[attach_id].copy()
    anchor_world = data.site_xpos[anchor_id].copy()
    np.testing.assert_allclose(attach_world, expected["attach_world"], atol=1e-9)
    np.testing.assert_allclose(anchor_world, expected["anchor_world"], atol=1e-9)

    delta = attach_world - anchor_world
    np.testing.assert_allclose(delta[list(expected["cross_axes"])], 0.0, atol=1e-12)
    assert model.tendon_range[tendon_id, 1] - model.tendon_length0[tendon_id] == pytest.approx(
        0.001, abs=1e-9
    )
    assert data.ten_length[tendon_id] == pytest.approx(model.tendon_length0[tendon_id], abs=1e-9)
    assert data.tendon_efcadr[tendon_id] == -1
    assert data.sensordata[model.sensor_adr[sensor_id]] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("axis", ("x", "y"))
def test_tendon_limit_pulls_toward_anchor_and_releases_toward_anchor(axis):
    expected = AXES[axis]
    model = load_model(expected["scene"])
    tendon_id = object_id(model, mujoco.mjtObj.mjOBJ_TENDON, expected["tendon"])
    sensor_id = object_id(model, mujoco.mjtObj.mjOBJ_SENSOR, expected["sensor"])
    world_joint_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "world_joint")
    qpos_axis = model.jnt_qposadr[world_joint_id] + expected["axis_index"]
    dof_axis = model.jnt_dofadr[world_joint_id] + expected["axis_index"]

    stretched = mujoco.MjData(model)
    stretched.qpos[qpos_axis] = 0.002
    mujoco.mj_forward(model, stretched)
    efc_address = stretched.tendon_efcadr[tendon_id]
    sensor_value = stretched.sensordata[model.sensor_adr[sensor_id]]
    assert stretched.ten_length[tendon_id] - expected["upper_limit"] == pytest.approx(
        0.001, abs=1e-9
    )
    assert efc_address >= 0
    assert sensor_value > 0.0
    assert sensor_value == pytest.approx(stretched.efc_force[efc_address], rel=1e-12)
    assert stretched.qfrc_constraint[dof_axis] < 0.0

    released = mujoco.MjData(model)
    released.qpos[qpos_axis] = -0.002
    mujoco.mj_forward(model, released)
    assert released.tendon_efcadr[tendon_id] == -1
    assert released.sensordata[model.sensor_adr[sensor_id]] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("axis", "variant", "iterations", "solref"), SENSITIVITY_SCENES
)
def test_solver_sensitivity_scene_changes_only_the_requested_solver_setting(
    axis, variant, iterations, solref
):
    scene = MODEL_DIR / f"rbkairos_scene_tendon_test_{axis}_{variant}.xml"
    model = load_model(scene)
    expected = AXES[axis]
    tendon_id = object_id(model, mujoco.mjtObj.mjOBJ_TENDON, expected["tendon"])

    assert model.ntendon == 1
    assert model.opt.cone == int(mujoco.mjtCone.mjCONE_ELLIPTIC)
    np.testing.assert_allclose(
        model.pair_solreffriction,
        np.tile([-1.0, -1.0], (model.npair, 1)),
    )
    assert model.opt.iterations == iterations
    np.testing.assert_allclose(model.tendon_solref_lim[tendon_id], solref, atol=1e-12)
    np.testing.assert_allclose(
        model.tendon_range[tendon_id], [0.0, expected["upper_limit"]], atol=1e-12
    )
    assert model.tendon_stiffness[tendon_id] == pytest.approx(0.0)
    assert model.tendon_damping[tendon_id] == pytest.approx(0.0)
    assert model.tendon_frictionloss[tendon_id] == pytest.approx(0.0)


def test_isolated_contact_force_meets_cv_and_solver_sensitivity_gates():
    for axis in ("x", "y"):
        measurements = {}
        for variant in ("baseline", "iterations100", "hard_limit"):
            scene = (
                AXES[axis]["scene"]
                if variant == "baseline"
                else MODEL_DIR / f"rbkairos_scene_tendon_test_{axis}_{variant}.xml"
            )
            median, coefficient_of_variation = simulate_isolated_contact_force(
                scene, axis
            )
            assert coefficient_of_variation <= 0.05
            measurements[variant] = median
        reference = measurements["baseline"]
        for variant in ("iterations100", "hard_limit"):
            assert abs(measurements[variant] - reference) / reference <= 0.03


def test_protected_friction_and_integral_limits_are_retained():
    root = ET.parse(BASE_MODEL).getroot()
    contact_pairs = {
        pair.get("name"): pair
        for pair in root.findall("./contact/pair")
        if pair.get("name") in {"fric_lf", "fric_rf", "fric_lb", "fric_rb"}
    }
    assert set(contact_pairs) == {"fric_lf", "fric_rf", "fric_lb", "fric_rb"}
    for pair in contact_pairs.values():
        assert [float(value) for value in pair.get("friction").split()] == [0.01, 0.7]
        assert [float(value) for value in pair.get("solreffriction").split()] == [
            -1.0,
            -1.0,
        ]

    urdf_root = ET.parse(PACKAGE_ROOT / "urdf" / "rbkairos.urdf").getroot()
    maximums = [
        float(param.text)
        for param in urdf_root.findall(".//param[@name='velocity_i_max']")
    ]
    minimums = [
        float(param.text)
        for param in urdf_root.findall(".//param[@name='velocity_i_min']")
    ]
    assert maximums == [1.0, 1.0, 1.0, 1.0]
    assert minimums == [-1.0, -1.0, -1.0, -1.0]
