import mujoco
from pathlib import Path

# change the path manually to fit your needs
MODEL_PATH = str(Path(__file__).resolve().parents[1] / "mujoco" / "rbkairos.xml")

QPOS_COMPONENTS = {
    mujoco.mjtJoint.mjJNT_FREE: ("x", "y", "z", "qw", "qx", "qy", "qz"),
    mujoco.mjtJoint.mjJNT_BALL: ("qw", "qx", "qy", "qz"),
    mujoco.mjtJoint.mjJNT_SLIDE: ("position",),
    mujoco.mjtJoint.mjJNT_HINGE: ("angle",),
}

JOINT_TYPE_NAMES = {
    mujoco.mjtJoint.mjJNT_FREE: "free",
    mujoco.mjtJoint.mjJNT_BALL: "ball",
    mujoco.mjtJoint.mjJNT_SLIDE: "slide",
    mujoco.mjtJoint.mjJNT_HINGE: "hinge",
}

model = mujoco.MjModel.from_xml_path(MODEL_PATH)

print(f"nq = {model.nq}")
print(f"njnt = {model.njnt}")
print()

for joint_id in range(model.njnt):
    joint_type = model.jnt_type[joint_id]
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    qpos_start = model.jnt_qposadr[joint_id]
    components = QPOS_COMPONENTS[joint_type]
    joint_type_name = JOINT_TYPE_NAMES[joint_type]

    print(f"{joint_name} ({joint_type_name} joint):")
    for offset, component in enumerate(components):
        qpos_index = qpos_start + offset
        print(f"  qpos[{qpos_index}] = {joint_name}.{component}")
