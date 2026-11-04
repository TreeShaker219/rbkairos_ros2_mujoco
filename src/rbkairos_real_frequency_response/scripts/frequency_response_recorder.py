#!/usr/bin/env python3

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


def now_string():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def number_token(value, digits=3):
    return f"{value:.{digits}f}".replace("-", "m").replace(".", "p")


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def dataclass_dict(args):
    return {key: value for key, value in vars(args).items()}


class FrequencyResponseRecorder:
    def __init__(self, args):
        self.args = args
        self.node = rclpy.create_node("rbkairos_frequency_response_recorder")
        self.cmd_pub = self.node.create_publisher(Twist, args.cmd_topic, 10)
        self.odom_sub = self.node.create_subscription(Odometry, args.odom_topic, self.odom_callback, 50)
        self.estop_sub = self.node.create_subscription(
            Bool, args.emergency_stop_topic, self.emergency_stop_callback, 10
        )

        self.latest_odom = None
        self.latest_odom_time = None
        self.emergency_stop = False
        self.emergency_stop_seen = False

        self.active_run = False
        self.run_start_s = 0.0
        self.cmd_records = []
        self.odom_records = []
        self.safety_events = []

    def now_s(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def odom_callback(self, msg):
        t_ros = self.now_s()
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        sample = {
            "t_ros": t_ros,
            "x": msg.pose.pose.position.x,
            "y": msg.pose.pose.position.y,
            "z": msg.pose.pose.position.z,
            "yaw": yaw,
            "vx": msg.twist.twist.linear.x,
            "vy": msg.twist.twist.linear.y,
            "wz": msg.twist.twist.angular.z,
        }
        self.latest_odom = sample
        self.latest_odom_time = t_ros
        if self.active_run:
            row = dict(sample)
            row["t_experiment"] = t_ros - self.run_start_s
            self.odom_records.append(row)

    def emergency_stop_callback(self, msg):
        self.emergency_stop = bool(msg.data)
        self.emergency_stop_seen = True

    def spin_for(self, duration_s):
        end_s = self.now_s() + max(0.0, duration_s)
        while rclpy.ok() and self.now_s() < end_s:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def publish_twist(self, vx=0.0, vy=0.0, wz=0.0):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)

    def publish_zero_for(self, duration_s):
        end_s = self.now_s() + max(0.0, duration_s)
        while rclpy.ok() and self.now_s() < end_s:
            self.publish_twist()
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def wait_until_ready(self):
        deadline_s = self.now_s() + self.args.startup_timeout
        while rclpy.ok() and self.now_s() < deadline_s:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            has_subscriber = self.cmd_pub.get_subscription_count() > 0
            has_odom = self.latest_odom_time is not None
            odom_fresh = has_odom and (self.now_s() - self.latest_odom_time) <= self.args.odom_timeout
            subscriber_ok = (not self.args.require_cmd_subscriber) or has_subscriber
            emergency_ok = (not self.args.require_emergency_stop) or self.emergency_stop_seen
            if subscriber_ok and odom_fresh and emergency_ok and not self.emergency_stop:
                return

        problems = []
        if self.args.require_cmd_subscriber and self.cmd_pub.get_subscription_count() == 0:
            problems.append(f"no subscriber on {self.args.cmd_topic}")
        if self.latest_odom_time is None:
            problems.append(f"no odom received on {self.args.odom_topic}")
        elif (self.now_s() - self.latest_odom_time) > self.args.odom_timeout:
            problems.append(f"odom on {self.args.odom_topic} is stale")
        if self.args.require_emergency_stop and not self.emergency_stop_seen:
            problems.append(f"no emergency stop state received on {self.args.emergency_stop_topic}")
        if self.emergency_stop:
            problems.append(f"{self.args.emergency_stop_topic} is true")
        raise RuntimeError("startup check failed: " + "; ".join(problems))

    def record_safety_event(self, event, detail):
        odom = self.latest_odom or {"x": math.nan, "y": math.nan, "yaw": math.nan}
        now = self.now_s()
        self.safety_events.append({
            "t_ros": now,
            "t_experiment": now - self.run_start_s if self.active_run else math.nan,
            "event": event,
            "detail": detail,
            "x": odom["x"],
            "y": odom["y"],
            "yaw": odom["yaw"],
        })

    def check_safety(self, initial_odom):
        now = self.now_s()
        if self.latest_odom_time is None or (now - self.latest_odom_time) > self.args.odom_timeout:
            return "odom_timeout", f"no fresh odom within {self.args.odom_timeout:.3f} s"
        if self.emergency_stop:
            return "emergency_stop", f"{self.args.emergency_stop_topic} is true"

        odom = self.latest_odom
        dx = odom["x"] - initial_odom["x"]
        dy = odom["y"] - initial_odom["y"]
        dyaw = normalize_angle(odom["yaw"] - initial_odom["yaw"])
        if abs(dx) > self.args.max_x:
            return "max_x", f"abs(dx)={abs(dx):.6f} > {self.args.max_x:.6f}"
        if abs(dy) > self.args.max_y:
            return "max_y", f"abs(dy)={abs(dy):.6f} > {self.args.max_y:.6f}"
        if abs(math.degrees(dyaw)) > self.args.max_yaw_deg:
            return "max_yaw", (
                f"abs(dyaw_deg)={abs(math.degrees(dyaw)):.6f} > {self.args.max_yaw_deg:.6f}"
            )
        return None, ""

    def run_frequency(self, experiment_dir, frequency_hz):
        run_dir = experiment_dir / "runs" / f"f_{number_token(frequency_hz)}Hz"
        run_dir.mkdir(parents=True, exist_ok=True)

        self.publish_zero_for(self.args.settle_between_frequencies)
        self.wait_until_ready()

        initial_odom = dict(self.latest_odom)
        self.cmd_records = []
        self.odom_records = []
        self.safety_events = []
        self.active_run = True
        self.run_start_s = self.now_s()

        omega = 2.0 * math.pi * frequency_hz
        duration_s = self.args.cycles / frequency_hz
        ramp_s = self.args.ramp_cycles / frequency_hz
        period_s = 1.0 / self.args.publish_rate
        stop_reason = "completed"

        next_publish_s = self.run_start_s
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.0)
            t_ros = self.now_s()
            t_experiment = t_ros - self.run_start_s
            if t_experiment >= duration_s:
                break

            event, detail = self.check_safety(initial_odom)
            if event:
                stop_reason = event
                self.record_safety_event(event, detail)
                break

            if t_ros >= next_publish_s:
                if ramp_s > 0.0:
                    ratio = max(0.0, min(1.0, t_experiment / ramp_s))
                    envelope = 0.5 - 0.5 * math.cos(math.pi * ratio)
                else:
                    envelope = 1.0
                phase_rad = omega * t_experiment
                cmd_vx = self.args.amplitude * envelope * math.sin(phase_rad)
                self.publish_twist(cmd_vx, 0.0, 0.0)
                self.cmd_records.append({
                    "t_ros": t_ros,
                    "t_experiment": t_experiment,
                    "frequency_hz": frequency_hz,
                    "omega_rad_s": omega,
                    "axis": self.args.axis,
                    "phase_rad": phase_rad,
                    "envelope": envelope,
                    "cmd_vx": cmd_vx,
                    "cmd_vy": 0.0,
                    "cmd_wz": 0.0,
                })
                next_publish_s += period_s

            sleep_s = max(0.0, min(0.02, next_publish_s - self.now_s()))
            rclpy.spin_once(self.node, timeout_sec=sleep_s)

        self.active_run = False
        self.publish_zero_for(self.args.post_stop_duration)

        self.write_run_outputs(run_dir, frequency_hz, initial_odom, stop_reason)
        return stop_reason == "completed"

    def write_run_outputs(self, run_dir, frequency_hz, initial_odom, stop_reason):
        omega = 2.0 * math.pi * frequency_hz
        run_id = f"f_{number_token(frequency_hz)}Hz"
        write_csv(
            run_dir / "cmd_vel.csv",
            [
                "t_ros", "t_experiment", "frequency_hz", "omega_rad_s", "axis", "phase_rad",
                "envelope", "cmd_vx", "cmd_vy", "cmd_wz",
            ],
            self.cmd_records,
        )
        write_csv(
            run_dir / "odom.csv",
            ["t_ros", "t_experiment", "x", "y", "z", "yaw", "vx", "vy", "wz"],
            self.odom_records,
        )
        write_csv(
            run_dir / "safety_events.csv",
            ["t_ros", "t_experiment", "event", "detail", "x", "y", "yaw"],
            self.safety_events,
        )

        rel_x = [row["x"] - initial_odom["x"] for row in self.odom_records]
        rel_y = [row["y"] - initial_odom["y"] for row in self.odom_records]
        rel_yaw = [normalize_angle(row["yaw"] - initial_odom["yaw"]) for row in self.odom_records]
        metadata = {
            "run_id": run_id,
            "environment": self.args.environment,
            "frequency_hz": frequency_hz,
            "omega_rad_s": omega,
            "amplitude": self.args.amplitude,
            "axis": self.args.axis,
            "cycles": self.args.cycles,
            "discard_cycles": self.args.discard_cycles,
            "ramp_cycles": self.args.ramp_cycles,
            "publish_rate": self.args.publish_rate,
            "stop_reason": stop_reason,
            "valid": stop_reason == "completed",
            "reject_reason": "" if stop_reason == "completed" else stop_reason,
            "sample_count_cmd": len(self.cmd_records),
            "sample_count_odom": len(self.odom_records),
            "max_abs_x": max([abs(value) for value in rel_x], default=0.0),
            "max_abs_y": max([abs(value) for value in rel_y], default=0.0),
            "max_abs_yaw_deg": max([abs(math.degrees(value)) for value in rel_yaw], default=0.0),
            "theoretical_position_amplitude": self.args.amplitude / omega,
            "controller_name": self.args.controller_name,
            "cmd_topic": self.args.cmd_topic,
            "odom_topic": self.args.odom_topic,
        }
        write_json(run_dir / "run_metadata.json", metadata)

    def destroy(self):
        self.publish_twist()
        self.node.destroy_node()


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path, data):
    with path.open("w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Record real RBKairos base frequency response data.")
    parser.add_argument("--environment", default="real_lab")
    parser.add_argument("--axis", choices=["x"], default="x")
    parser.add_argument("--amplitude", type=float, default=0.05)
    parser.add_argument(
        "--frequencies",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.50, 0.80, 1.20, 2.00],
    )
    parser.add_argument("--publish-rate", type=float, default=50.0)
    parser.add_argument("--cycles", type=float, default=6.0)
    parser.add_argument("--discard-cycles", type=float, default=2.0)
    parser.add_argument("--ramp-cycles", type=float, default=1.0)
    parser.add_argument("--cmd-topic", default="/base_cmd_vel")
    parser.add_argument("--odom-topic", default="/rbkairos_base/odom")
    parser.add_argument("--emergency-stop-topic", default="/emergency_stop")
    parser.add_argument(
        "--require-cmd-subscriber",
        dest="require_cmd_subscriber",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-require-cmd-subscriber",
        dest="require_cmd_subscriber",
        action="store_false",
    )
    parser.add_argument(
        "--require-emergency-stop",
        dest="require_emergency_stop",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-require-emergency-stop",
        dest="require_emergency_stop",
        action="store_false",
    )
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--odom-timeout", type=float, default=0.3)
    parser.add_argument("--max-x", type=float, default=0.35)
    parser.add_argument("--max-y", type=float, default=0.06)
    parser.add_argument("--max-yaw-deg", type=float, default=5.0)
    parser.add_argument("--post-stop-duration", type=float, default=0.5)
    parser.add_argument("--settle-between-frequencies", type=float, default=1.0)
    parser.add_argument("--output-dir", default="data/frequency_response/real_lab")
    parser.add_argument("--controller-name", default="rbkairos_base_twist_controller")
    args = parser.parse_args(argv)

    if args.amplitude <= 0.0:
        parser.error("--amplitude must be positive")
    if args.publish_rate <= 0.0:
        parser.error("--publish-rate must be positive")
    if args.cycles <= 0.0:
        parser.error("--cycles must be positive")
    if any(freq <= 0.0 for freq in args.frequencies):
        parser.error("--frequencies must all be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    experiment_name = (
        f"{now_string()}_{args.environment}_{args.axis}_A{number_token(args.amplitude, digits=2)}"
    )
    experiment_dir = Path(args.output_dir) / experiment_name
    (experiment_dir / "runs").mkdir(parents=True, exist_ok=True)

    rclpy.init(args=None)
    recorder = FrequencyResponseRecorder(args)
    success = True
    try:
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "experiment_dir": str(experiment_dir),
            "arguments": dataclass_dict(args),
            "frequencies_hz": args.frequencies,
        }
        write_json(experiment_dir / "metadata.json", metadata)

        recorder.wait_until_ready()
        for frequency_hz in args.frequencies:
            run_ok = recorder.run_frequency(experiment_dir, frequency_hz)
            success = success and run_ok
            if not run_ok:
                break
    except KeyboardInterrupt:
        success = False
        recorder.publish_zero_for(args.post_stop_duration)
    except Exception as exc:
        success = False
        recorder.publish_zero_for(args.post_stop_duration)
        print(f"frequency_response_recorder failed: {exc}", file=sys.stderr)
    finally:
        recorder.destroy()
        rclpy.shutdown()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
