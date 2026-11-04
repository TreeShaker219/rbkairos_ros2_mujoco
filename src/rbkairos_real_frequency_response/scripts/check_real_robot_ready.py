#!/usr/bin/env python3

import argparse
import sys

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


class ReadinessChecker:
    def __init__(self, args):
        self.args = args
        self.node = rclpy.create_node("rbkairos_frequency_response_ready_check")
        self.cmd_pub = self.node.create_publisher(Twist, args.cmd_topic, 10)
        self.odom_seen = False
        self.emergency_seen = False
        self.emergency_stop = None
        self.node.create_subscription(Odometry, args.odom_topic, self.odom_callback, 10)
        self.node.create_subscription(Bool, args.emergency_stop_topic, self.emergency_callback, 10)

    def odom_callback(self, msg):
        self.odom_seen = True

    def emergency_callback(self, msg):
        self.emergency_seen = True
        self.emergency_stop = bool(msg.data)

    def run(self):
        end_time = self.node.get_clock().now().nanoseconds * 1e-9 + self.args.timeout
        while rclpy.ok() and self.node.get_clock().now().nanoseconds * 1e-9 < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.done():
                break

        failures = []
        if self.args.require_cmd_subscriber and self.cmd_pub.get_subscription_count() == 0:
            failures.append(f"{self.args.cmd_topic} has no subscriber")
        if not self.odom_seen:
            failures.append(f"{self.args.odom_topic} did not publish nav_msgs/msg/Odometry")
        if self.args.require_emergency_stop and not self.emergency_seen:
            failures.append(f"{self.args.emergency_stop_topic} did not publish std_msgs/msg/Bool")
        if self.emergency_stop is True:
            failures.append(f"{self.args.emergency_stop_topic} is true")

        print(f"cmd_topic: {self.args.cmd_topic}")
        print(f"cmd_subscriptions: {self.cmd_pub.get_subscription_count()}")
        print(f"odom_topic: {self.args.odom_topic}")
        print(f"odom_seen: {self.odom_seen}")
        print(f"emergency_stop_topic: {self.args.emergency_stop_topic}")
        print(f"emergency_stop_seen: {self.emergency_seen}")
        print(f"emergency_stop_value: {self.emergency_stop}")

        if failures:
            print("ready: false")
            for failure in failures:
                print(f"failure: {failure}")
            return 1

        print("ready: true")
        return 0

    def done(self):
        cmd_ok = (not self.args.require_cmd_subscriber) or self.cmd_pub.get_subscription_count() > 0
        estop_ok = (not self.args.require_emergency_stop) or self.emergency_seen
        return cmd_ok and self.odom_seen and estop_ok


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Check readiness for RBKairos real frequency response tests.")
    parser.add_argument("--cmd-topic", default="/base_cmd_vel")
    parser.add_argument("--odom-topic", default="/rbkairos_base/odom")
    parser.add_argument("--emergency-stop-topic", default="/emergency_stop")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--require-cmd-subscriber", action="store_true")
    parser.add_argument("--require-emergency-stop", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rclpy.init(args=None)
    checker = ReadinessChecker(args)
    try:
        return checker.run()
    finally:
        checker.node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
