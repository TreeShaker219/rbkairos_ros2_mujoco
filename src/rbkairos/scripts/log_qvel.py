#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Log MuJoCo generalized velocities (qvel) and accelerations (qacc) to a CSV file.

Usage examples:
  python3 log_qvel.py
  python3 log_qvel.py --xml src/rbkairos/mujoco/base.xml --duration 20 --hz 200
  python3 log_qvel.py --ctrl 5 5 5 5 --kp 2.0
  python3 log_qvel.py --wheel-joints a b c d --actuators m1 m2 m3 m4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional

import mujoco
import numpy as np


def _unique_names(names: List[str]) -> List[str]:
    """Make header names unique (MuJoCo may repeat joint names across multiple DOFs)."""
    seen = {}
    out = []
    for n in names:
        if n not in seen:
            seen[n] = 0
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}#{seen[n]}")
    return out


def _mj_name2id_or_raise(m: mujoco.MjModel, objtype: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(m, objtype, name)
    if idx < 0:
        raise ValueError(f"MuJoCo model 中找不到 {objtype.name} 名称: {name}")
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Log qvel & qacc from a MuJoCo model to CSV.")

    parser.add_argument("--xml", default="src/rbkairos/mujoco/base.xml",
                        help="MJCF/XML path")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="simulation duration in seconds")
    parser.add_argument("--hz", type=float, default=100.0,
                        help="logging sample frequency (Hz)")
    parser.add_argument("--kp", type=float, default=2.0,
                        help="P gain for wheel speed tracking (set 0 for no control)")

    # wheel joints (used to find qvel/qacc indices)
    parser.add_argument("--wheel-joints", nargs=4, default=[
        "left_front_wheel_joint",
        "right_front_wheel_joint",
        "left_back_wheel_joint",
        "right_back_wheel_joint",
    ], help="4 joint names for wheels (LF RF LB RB)")

    # actuators (optional). If omitted, defaults to same names as wheel joints.
    parser.add_argument("--actuators", nargs=4, default=None,
                        help="4 actuator names (LF RF LB RB). Default: same as wheel-joints")

    # desired wheel speed pattern
    parser.add_argument("--ctrl", nargs=4, type=float, default=[0.0, 0.0, 0.0, 0.0],
                        help="desired wheel angular velocities (LF RF LB RB)")

    parser.add_argument("--out", default=None,
                        help="output CSV path. Default: data/qvel_qacc_T{duration}_Hz{hz}.csv")

    args = parser.parse_args()

    xml_path = Path(args.xml)
    if not xml_path.exists():
        raise FileNotFoundError(f"找不到 xml: {xml_path}")

    duration: float = float(args.duration)
    sample_hz: float = float(args.hz)
    kp: float = float(args.kp)
    ctrl_pattern = np.array(args.ctrl, dtype=float)

    # Load model
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)

    # Build DOF -> joint-name list (header). Might contain duplicates; make unique for CSV.
    dof_names = []
    for i in range(m.nv):
        j = m.dof_jntid[i]
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        dof_names.append(jname if jname is not None else f"joint_{j}")
    dof_names = _unique_names(dof_names)

    # Wheel DOF indices (qvel/qacc index)
    wheel_joints: List[str] = list(args.wheel_joints)
    wheel_dof: List[int] = []
    for jname in wheel_joints:
        jid = _mj_name2id_or_raise(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        dof_adr = int(m.jnt_dofadr[jid])
        wheel_dof.append(dof_adr)

    # Actuator indices
    actuator_names: List[str] = list(args.actuators) if args.actuators is not None else wheel_joints
    act_ids: List[int] = []
    for aname in actuator_names:
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        if aid < 0:
            raise ValueError(
                f"找不到 actuator: {aname}\n"
                f"提示：很多模型里 actuator 的 name 不等于 joint 的 name。\n"
                f"请用 --actuators 指定正确的4个 actuator 名称。"
            )
        act_ids.append(aid)

    dt = float(m.opt.timestep)
    if dt <= 0:
        raise ValueError(f"非法 timestep: {dt}")

    steps_total = int(np.ceil(duration / dt))

    # Avoid modulo by 0 and make sampling stable
    # Desired period in steps = 1/(hz*dt)
    if sample_hz <= 0:
        raise ValueError("--hz 必须 > 0")
    sample_every = max(1, int(round(1.0 / (sample_hz * dt))))

    # Output path
    if args.out is None:
        out_dir = Path("data")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"qvel_qacc_T{duration:g}_Hz{sample_hz:g}.csv"
    else:
        out_csv = Path(args.out)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    # Simulation loop
    for step in range(steps_total):
        # Simple P controller to track wheel speed (set kp=0 to disable)
        if kp != 0.0:
            for k, aid in enumerate(act_ids):
                vel = float(d.qvel[wheel_dof[k]])
                d.ctrl[aid] = kp * (ctrl_pattern[k] - vel)

        mujoco.mj_step(m, d)

        if step % sample_every == 0:
            t = float(d.time)  # use MuJoCo's time for better accuracy
            rows.append([t, *d.qvel.tolist(), *d.qacc.tolist()])

    # Write CSV
    header = (
        ["time"]
        + [f"qvel_{n}" for n in dof_names]
        + [f"qacc_{n}" for n in dof_names]
    )

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"写入 {out_csv}，行数: {len(rows)}，dt={dt:g}s，sample_every={sample_every} steps")


if __name__ == "__main__":
    main()
