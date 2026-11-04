# joint_type has all these possibilities:
# - JointModelR[*]: Revolute Joint, with rotation axis [*] in [X,Y,Z]"
# - JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] in [X,Y,Z]"
# - JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z"
# - JointModelFreeFlyer: Joint enabling 3D rotation and translations.")
# details in website https://docs.ros.org/en/latest-available/api/pinocchio/html/joint_8hpp_source.html


import pinocchio as pin


def q_component_names(joint_type, nq):
    """Convert the q components used by one joint into readable names."""
    if joint_type == "JointModelFreeFlyer" and nq == 7:
        return ["x", "y", "z", "qx", "qy", "qz", "qw"]

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

        # shortname() converts the URDF/MJCF joint type into Pinocchio's joint type string.
        joint_type = model.joints[joint_id].shortname()
        components = q_component_names(joint_type, nq)
        for offset, component in enumerate(components):
            q_index = q_start + offset
            q_name = f"{joint_name}.{component}"
            print(
                f"  q[{q_index:2d}] = {q_name:<38} "
                f"({joint_type:<29}) value={q[q_index]: .6g}"
            )


def test_urdf_match_with_mjcf(model):
    print("model name: " + model.name)
    print("nq: " + str(model.nq))
    print("lowerLimits: " + str(model.lowerPositionLimit))
    print("upperLimits: " + str(model.upperPositionLimit))
    
    data = model.createData()
    q = pin.neutral(model)
    q_list = q.tolist()
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
    test_urdf_match_with_mjcf(model_mjcf)
    print("")
    print("urdf test")
    model_urdf = pin.buildModelFromUrdf("/home/treeshaker/ba_zeng/src/rbkairos/urdf/rbkairos.urdf")
    test_urdf_match_with_mjcf(model_urdf)
