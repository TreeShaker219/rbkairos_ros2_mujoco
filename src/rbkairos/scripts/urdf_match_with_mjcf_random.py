import pinocchio as pin
import math

import numpy as np


# Preset a fixed "random" pose so every run is reproducible.
TARGET_BASE_WORLD_POSITION = (0.30, -0.18, 0.60)
TARGET_BASE_YAW = 0.35

# Preset angles for the four actuated wheels, in radians.
WHEEL_ANGLES = {
    "left_front_wheel_joint": 0.40,
    "right_front_wheel_joint": -0.35,
    "left_back_wheel_joint": 0.70,
    "right_back_wheel_joint": -0.55,
}

# Preset angles for the six UR5 joints, in radians.
UR5_ANGLES = {
    "ur5/shoulder_pan_joint": 0.20,
    "ur5/shoulder_lift_joint": -0.45,
    "ur5/elbow_joint": 0.65,
    "ur5/wrist_1_joint": -0.35,
    "ur5/wrist_2_joint": 0.25,
    "ur5/wrist_3_joint": -0.15,
}


def yaw_quaternion(yaw):
    """Return [qx, qy, qz, qw] for a Pinocchio free-flyer."""
    half_yaw = 0.5 * yaw
    return [0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)]


def continuous_joint_q(theta):
    """Represent a URDF continuous joint as [cos(theta), sin(theta)] in Pinocchio."""
    return [math.cos(theta), math.sin(theta)]


def planar_joint_q(yaw):
    """Represent a planar joint rotation as [cos(yaw), sin(yaw)] in Pinocchio."""
    return [math.cos(yaw), math.sin(yaw)]


def preset_mjcf_q_list():
    """Build the 21-dimensional Pinocchio q vector for the MJCF model."""
    x, y, z = TARGET_BASE_WORLD_POSITION
    qx, qy, qz, qw = yaw_quaternion(TARGET_BASE_YAW)

    # Keep z at 0 because the URDF root is planar and has no z degree of freedom.
    root_z = 0.0

    return [
        x, y, root_z, qx, qy, qz, qw,
        WHEEL_ANGLES["right_front_wheel_joint"],
        WHEEL_ANGLES["left_front_wheel_joint"],
        WHEEL_ANGLES["left_back_wheel_joint"],
        WHEEL_ANGLES["right_back_wheel_joint"],
        WHEEL_ANGLES["left_front_wheel_joint"],
        WHEEL_ANGLES["right_front_wheel_joint"],
        WHEEL_ANGLES["left_back_wheel_joint"],
        WHEEL_ANGLES["right_back_wheel_joint"],
        UR5_ANGLES["ur5/shoulder_pan_joint"],
        UR5_ANGLES["ur5/shoulder_lift_joint"],
        UR5_ANGLES["ur5/elbow_joint"],
        UR5_ANGLES["ur5/wrist_1_joint"],
        UR5_ANGLES["ur5/wrist_2_joint"],
        UR5_ANGLES["ur5/wrist_3_joint"],
    ]


def preset_urdf_q_list():
    """Build the 18-dimensional Pinocchio q vector for the URDF model."""
    x, y, _ = TARGET_BASE_WORLD_POSITION

    return [
        x, y, *planar_joint_q(TARGET_BASE_YAW),
        *continuous_joint_q(WHEEL_ANGLES["left_back_wheel_joint"]),
        *continuous_joint_q(WHEEL_ANGLES["left_front_wheel_joint"]),
        *continuous_joint_q(WHEEL_ANGLES["right_back_wheel_joint"]),
        *continuous_joint_q(WHEEL_ANGLES["right_front_wheel_joint"]),
        UR5_ANGLES["ur5/shoulder_pan_joint"],
        UR5_ANGLES["ur5/shoulder_lift_joint"],
        UR5_ANGLES["ur5/elbow_joint"],
        UR5_ANGLES["ur5/wrist_1_joint"],
        UR5_ANGLES["ur5/wrist_2_joint"],
        UR5_ANGLES["ur5/wrist_3_joint"],
    ]


def q_component_names(joint_type, nq):
    """Convert the q components used by one joint into readable names."""
    if joint_type == "JointModelFreeFlyer" and nq == 7:
        return ["x", "y", "z", "qx", "qy", "qz", "qw"]

    if joint_type == "JointModelPlanar" and nq == 4:
        return ["x", "y", "cos(theta)", "sin(theta)"]

    if joint_type.startswith("JointModelRUB") and nq == 2:
        return ["cos(theta)", "sin(theta)"]

    if nq == 1:
        return ["theta"]

    return [f"component_{i}" for i in range(nq)]


def print_q_names(model, q):
    """Print the meaning of each component in the Pinocchio q vector."""
    print("q names (Pinocchio order):")
    for joint_id, joint_name in enumerate(model.names):
        q_start = model.idx_qs[joint_id]
        nq = model.nqs[joint_id]
        if nq == 0:
            continue

        joint_type = model.joints[joint_id].shortname()
        components = q_component_names(joint_type, nq)
        for offset, component in enumerate(components):
            q_index = q_start + offset
            q_name = f"{joint_name}.{component}"
            print(
                f"  q[{q_index:2d}] = {q_name:<38} "
                f"({joint_type:<29}) value={q[q_index]: .6g}"
            )


def test_urdf_match_with_mjcf(model, q_list):
    print("model name: " + model.name)
    print("nq: " + str(model.nq))
    print("lowerLimits: " + str(model.lowerPositionLimit))
    print("upperLimits: " + str(model.upperPositionLimit))
    
    data = model.createData()
    q = np.array(q_list, dtype=np.float64)
    if q.shape[0] != model.nq:
        raise ValueError(f"q_list length {q.shape[0]} does not match model.nq {model.nq}")

    print("q_list: " + str(q_list))
    print_q_names(model, q)
    # Perform the forward kinematics
    pin.forwardKinematics(model, data, q)
    # Print each joint's kinematic position xyz and orientation rpy in the world frame.
    for name, oMi in zip(model.names, data.oMi):
        rpy = pin.rpy.matrixToRpy(oMi.rotation)
        print(
            "{:<24} : xyz=({: .2f} {: .2f} {: .2f}) "
            "rpy=({: .2f} {: .2f} {: .2f})".format(
                name,
                *oMi.translation.T.flat,
                *rpy.T.flat,
            )
        )

if __name__ == '__main__':
    print("mjcf test")
    model_mjcf = pin.RobotWrapper.BuildFromMJCF("/home/treeshaker/ba_zeng/src/rbkairos/mujoco/rbkairos.xml").model
    test_urdf_match_with_mjcf(model_mjcf, preset_mjcf_q_list())
    print("")
    print("urdf test")
    model_urdf = pin.buildModelFromUrdf("/home/treeshaker/ba_zeng/src/rbkairos/urdf/rbkairos.urdf")
    test_urdf_match_with_mjcf(model_urdf, preset_urdf_q_list())
