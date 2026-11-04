#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Send preset UR5 arm joint configurations through JointTrajectoryController.

Expected setup:
  ros2 launch rbkairos mujoco_mecanum_drive.launch.py

Examples:
  python3 src/rbkairos/scripts/arm_preset_commander.py --preset forward
  python3 src/rbkairos/scripts/arm_preset_commander.py --preset left --duration 3.0
  ros2 run rbkairos arm_preset_commander.py --preset folded --duration 2.0
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Optional, Tuple

import rclpy
from builtin_interfaces.msg import Duration, Time
from mujoco_ros2_control_msgs.msg import MujocoDofState
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ARM_JOINT_NAMES = [
    "ur5/shoulder_pan_joint",
    "ur5/shoulder_lift_joint",
    "ur5/elbow_joint",
    "ur5/wrist_1_joint",
    "ur5/wrist_2_joint",
    "ur5/wrist_3_joint",
]

DEFAULT_COMMAND_TOPIC = "/ur5_joint_trajectory_controller/joint_trajectory"
DEFAULT_STATE_TOPIC = "/mujoco_dof_state"

HALF_PI = math.pi / 2.0

# Base frame convention here is +X forward and +Y left.
presets = {
    "folded": [0.0, -HALF_PI, HALF_PI, -HALF_PI, -HALF_PI, 0.0],
    "forward": [HALF_PI, -0.8, 1.2, -0.4, 0.0, 0.0],
    "left": [math.pi, -0.8, 1.2, -0.4, 0.0, 0.0],
    "right": [0.0, -0.8, 1.2, -0.4, 0.0, 0.0],
    "up": [0.0, -HALF_PI, 0.0, -HALF_PI, 0.0, 0.0],
}


def seconds_to_duration(seconds: float) -> Duration:
    whole_seconds = int(seconds)
    nanoseconds = int(round((seconds - whole_seconds) * 1_000_000_000))
    if nanoseconds >= 1_000_000_000:
        whole_seconds += 1
        nanoseconds -= 1_000_000_000

    msg = Duration()
    msg.sec = whole_seconds
    msg.nanosec = nanoseconds
    return msg


def format_positions(names: list[str], positions: list[float]) -> str:
    return ", ".join(
        f"{name}={position:.6f}" for name, position in zip(names, positions)
    )


class ArmPresetCommander(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("arm_preset_commander")
        self.args = args
        self.latest_arm_positions: Optional[list[float]] = None

        self.publisher = self.create_publisher(
            JointTrajectory, args.command_topic, 10
        )
        self.state_subscription = self.create_subscription(
            MujocoDofState,
            args.state_topic,
            self._on_mujoco_dof_state,
            20,
        )

    def _on_mujoco_dof_state(self, msg: MujocoDofState) -> None:
        name_to_index = {name: index for index, name in enumerate(msg.dof_names)}
        if not all(name in name_to_index for name in ARM_JOINT_NAMES):
            return

        positions = list(msg.qpos)
        arm_positions = []
        for name in ARM_JOINT_NAMES:
            index = name_to_index[name]
            if index >= len(positions):
                return
            arm_positions.append(positions[index])

        self.latest_arm_positions = arm_positions

    def wait_for_controller(self) -> None:
        deadline = time.monotonic() + self.args.startup_timeout
        while rclpy.ok() and self.publisher.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                if self.args.positions is not None:
                    raise RuntimeError(
                        f"No controller subscriber on {self.args.command_topic}"
                    )
                self.get_logger().warn(
                    "No subscriber detected on "
                    f"{self.args.command_topic}; publishing anyway."
                )
                return
            rclpy.spin_once(self, timeout_sec=0.1)

    def publish_preset(self, preset_name: str, positions: list[float]) -> None:
        trajectory = JointTrajectory()
        # A zero stamp tells JointTrajectoryController to start immediately.
        # Using wall-clock time breaks when the controller runs on simulation time.
        trajectory.header.stamp = Time()
        trajectory.joint_names = list(ARM_JOINT_NAMES)

        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.time_from_start = seconds_to_duration(self.args.duration)
        trajectory.points = [point]

        self.publisher.publish(trajectory)
        self.get_logger().info(
            f"Published preset '{preset_name}' to {self.args.command_topic}"
        )

    def wait_for_motion(self) -> None:
        deadline = time.monotonic() + self.args.duration + self.args.settle_time
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)


def positive_float(parser: argparse.ArgumentParser, name: str, value: float) -> None:
    if value <= 0.0:
        parser.error(f"{name} must be > 0")


def nonnegative_float(parser: argparse.ArgumentParser, name: str, value: float) -> None:
    if value < 0.0:
        parser.error(f"{name} must be >= 0")


def parse_args(argv: Optional[list[str]] = None) -> Tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Send a preset UR5 arm pose to the joint trajectory controller."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(presets.keys()),
        default="forward",
        help="arm preset to command",
    )
    parser.add_argument(
        "--positions", type=float, nargs=6,
        help="explicit joint targets in ARM_JOINT_NAMES order; overrides --preset",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="trajectory time_from_start in seconds",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.5,
        help="extra seconds to wait after the requested duration",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=5.0,
        help="seconds to wait for the controller command subscriber",
    )
    parser.add_argument(
        "--command-topic",
        default=DEFAULT_COMMAND_TOPIC,
        help="JointTrajectory command topic for the arm controller",
    )
    parser.add_argument(
        "--state-topic",
        default=DEFAULT_STATE_TOPIC,
        help="MujocoDofState topic used to read final simulated joint positions",
    )

    args, ros_args = parser.parse_known_args(argv)
    if ros_args:
        if "--ros-args" not in ros_args:
            parser.error("unknown arguments: " + " ".join(ros_args))
        ros_arg_index = ros_args.index("--ros-args")
        if ros_arg_index > 0:
            parser.error("unknown arguments: " + " ".join(ros_args[:ros_arg_index]))
        ros_args = ros_args[ros_arg_index:]

    if args.positions is not None and not all(math.isfinite(x) for x in args.positions):
        parser.error("--positions must contain six finite numbers")
    positive_float(parser, "--duration", args.duration)
    nonnegative_float(parser, "--settle-time", args.settle_time)
    nonnegative_float(parser, "--startup-timeout", args.startup_timeout)

    return args, ros_args


def main(argv: Optional[list[str]] = None) -> None:
    args, ros_args = parse_args(argv)
    selected_positions = (
        list(args.positions) if args.positions is not None else list(presets[args.preset])
    )
    target_label = "URDF initial hold" if args.positions is not None else args.preset

    rclpy.init(args=ros_args)
    node = ArmPresetCommander(args)
    try:
        node.wait_for_controller()
        node.publish_preset(target_label, selected_positions)
        node.wait_for_motion()

        print(f"Final preset: {target_label}")
        print("Target joint positions:")
        print(format_positions(ARM_JOINT_NAMES, selected_positions))
        if node.latest_arm_positions is not None:
            print("Measured final joint positions:")
            print(format_positions(ARM_JOINT_NAMES, node.latest_arm_positions))
        else:
            print(f"No measured final positions received from {args.state_topic}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
