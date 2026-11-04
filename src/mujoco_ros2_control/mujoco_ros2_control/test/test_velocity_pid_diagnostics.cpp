// Copyright (c) 2026 OpenAI
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

#include "control_toolbox/pid.hpp"
#include "mujoco_ros2_control/velocity_pid_diagnostics.hpp"

namespace
{
double clamp_effort(double value, double limit)
{
  return std::max(-limit, std::min(value, limit));
}
}  // namespace

TEST(VelocityPidDiagnostics, PreservesTheExistingControllerCommandSequence)
{
  control_toolbox::Pid baseline(2.0, 0.4, 0.2, 1.0, -1.0, true);
  control_toolbox::Pid instrumented(2.0, 0.4, 0.2, 1.0, -1.0, true);
  constexpr double effort_limit = 5.0;
  constexpr std::uint64_t period_ns = 10000000;
  const std::vector<double> errors{0.2, 0.4, -0.1, 3.0, -3.0};

  for (const double error : errors) {
    const double baseline_raw = baseline.computeCommand(error, period_ns);
    const double baseline_clamped = clamp_effort(baseline_raw, effort_limit);
    const auto diagnostic = mujoco_ros2_control::compute_velocity_pid_diagnostic(
      instrumented, "wheel", error + 0.25, 0.25, period_ns, true, effort_limit);

    EXPECT_DOUBLE_EQ(diagnostic.pid_raw_output, baseline_raw);
    EXPECT_DOUBLE_EQ(diagnostic.effort_clamped_output, baseline_clamped);
    EXPECT_NEAR(
      diagnostic.p_term + diagnostic.i_term + diagnostic.d_term,
      diagnostic.pid_raw_output, 1e-12);
  }
}

TEST(VelocityPidDiagnostics, ReportsIntegralAndEffortClampStatesIndependently)
{
  constexpr std::uint64_t period_ns = 100000000;

  control_toolbox::Pid integral_limited(0.0, 10.0, 0.0, 0.1, -0.1, true);
  const auto positive_i = mujoco_ros2_control::compute_velocity_pid_diagnostic(
    integral_limited, "wheel", 1.0, 0.0, period_ns, true, 100.0);
  EXPECT_TRUE(positive_i.i_clamp_active);
  EXPECT_DOUBLE_EQ(positive_i.candidate_i_term, 1.0);
  EXPECT_FALSE(positive_i.effort_clamp_active);
  EXPECT_NEAR(positive_i.i_term, 0.1, 1e-12);

  control_toolbox::Pid integral_limited_without_antiwindup(
    0.0, 10.0, 0.0, 0.1, -0.1, false);
  const auto positive_i_without_antiwindup =
    mujoco_ros2_control::compute_velocity_pid_diagnostic(
    integral_limited_without_antiwindup, "wheel", 1.0, 0.0, period_ns, true, 100.0);
  EXPECT_TRUE(positive_i_without_antiwindup.i_clamp_active);
  EXPECT_FALSE(positive_i_without_antiwindup.effort_clamp_active);
  EXPECT_NEAR(positive_i_without_antiwindup.i_term, 0.1, 1e-12);

  control_toolbox::Pid effort_limited(10.0, 0.0, 0.0, 1.0, -1.0, true);
  const auto positive_effort = mujoco_ros2_control::compute_velocity_pid_diagnostic(
    effort_limited, "wheel", 2.0, 0.0, period_ns, true, 5.0);
  EXPECT_FALSE(positive_effort.i_clamp_active);
  EXPECT_TRUE(positive_effort.effort_clamp_active);
  EXPECT_DOUBLE_EQ(positive_effort.pid_raw_output, 20.0);
  EXPECT_DOUBLE_EQ(positive_effort.effort_clamped_output, 5.0);

  control_toolbox::Pid negative_effort_limited(10.0, 0.0, 0.0, 1.0, -1.0, true);
  const auto negative_effort = mujoco_ros2_control::compute_velocity_pid_diagnostic(
    negative_effort_limited, "wheel", -2.0, 0.0, period_ns, true, 5.0);
  EXPECT_TRUE(negative_effort.effort_clamp_active);
  EXPECT_DOUBLE_EQ(negative_effort.effort_clamped_output, -5.0);
}

TEST(VelocityPidDiagnostics, LatchesControllerRateClampAndCandidatePeakEvidence)
{
  constexpr std::uint64_t period_ns = 100000000;
  control_toolbox::Pid pid(0.0, 10.0, 0.0, 0.1, -0.1, true);

  auto first = mujoco_ros2_control::compute_velocity_pid_diagnostic(
    pid, "wheel", 1.0, 0.0, period_ns, true, 100.0);
  first = mujoco_ros2_control::accumulate_velocity_pid_evidence({}, first);
  EXPECT_EQ(first.control_update_count, 1u);
  EXPECT_TRUE(first.i_clamp_ever);
  EXPECT_FALSE(first.effort_clamp_ever);
  EXPECT_DOUBLE_EQ(first.peak_abs_candidate_i_term, 1.0);

  auto second = mujoco_ros2_control::compute_velocity_pid_diagnostic(
    pid, "wheel", 0.0, 0.0, period_ns, true, 100.0);
  second = mujoco_ros2_control::accumulate_velocity_pid_evidence(first, second);
  EXPECT_EQ(second.control_update_count, 2u);
  EXPECT_TRUE(second.i_clamp_ever);
  EXPECT_FALSE(second.effort_clamp_ever);
  EXPECT_DOUBLE_EQ(second.peak_abs_candidate_i_term, 1.0);
}

TEST(VelocityPidDiagnostics, PublishesActualRuntimeGainsAndSignals)
{
  control_toolbox::Pid pid(3.0, 0.5, 0.1, 0.8, -0.7, true);
  const auto diagnostic = mujoco_ros2_control::compute_velocity_pid_diagnostic(
    pid, "left_front_wheel_joint", 1.25, 0.5, 10000000, true, 42.0);

  EXPECT_TRUE(diagnostic.valid);
  EXPECT_EQ(diagnostic.joint_name, "left_front_wheel_joint");
  EXPECT_DOUBLE_EQ(diagnostic.velocity_target, 1.25);
  EXPECT_DOUBLE_EQ(diagnostic.velocity_actual, 0.5);
  EXPECT_DOUBLE_EQ(diagnostic.velocity_error, 0.75);
  EXPECT_DOUBLE_EQ(diagnostic.kp, 3.0);
  EXPECT_DOUBLE_EQ(diagnostic.ki, 0.5);
  EXPECT_DOUBLE_EQ(diagnostic.kd, 0.1);
  EXPECT_DOUBLE_EQ(diagnostic.i_max, 0.8);
  EXPECT_DOUBLE_EQ(diagnostic.i_min, -0.7);
  EXPECT_TRUE(diagnostic.antiwindup);
  EXPECT_DOUBLE_EQ(diagnostic.effort_limit, 42.0);
}
