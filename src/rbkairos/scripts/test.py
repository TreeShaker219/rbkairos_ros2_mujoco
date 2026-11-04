import mujoco
import numpy as np
import csv

xml_path = "src/rbkairos/mujoco/base.xml"
duration   = 10.0      # 秒
sample_hz  = 100       # 采样频率
ctrl_pattern = [0, 0, 0, 0]  # 4 个轮子的控制指令（根据需要改）
Kp = 2.0               # 简单 PD 把轮速跟到 ctrl_pattern

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

# 关节/DOF 名称映射
dof_names = []
for i in range(m.nv):
    j = m.dof_jntid[i]
    dof_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))

# 轮子的 dof 索引和 actuator 索引
wheel_joints = [
    "left_front_wheel_joint",
    "right_front_wheel_joint",
    "left_back_wheel_joint",
    "right_back_wheel_joint",
]
wheel_dof = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
             for j in wheel_joints]
act_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
           for j in wheel_joints]

dt = m.opt.timestep
steps_total = int(duration / dt)
sample_every = int(round(1.0 / (sample_hz * dt)))

rows = []
for step in range(steps_total):
    # 简单 PD 设定轮速（如需纯观察自由漂移，把 ctrl_pattern 置 0 即可）
    for k, aid in enumerate(act_ids):
        vel = d.qvel[wheel_dof[k]]
        d.ctrl[aid] = Kp * (ctrl_pattern[k] - vel)

    mujoco.mj_step(m, d)

    if step % sample_every == 0:
        t = step * dt
        rows.append([t, *d.qvel.tolist()])

# 写 CSV
with open("qvel_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time"] + dof_names)
    writer.writerows(rows)

print("写入 qvel_log.csv，行数:", len(rows))