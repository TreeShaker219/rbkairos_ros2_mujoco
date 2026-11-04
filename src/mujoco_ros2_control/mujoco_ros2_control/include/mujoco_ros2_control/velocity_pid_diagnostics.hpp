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

#ifndef MUJOCO_ROS2_CONTROL__VELOCITY_PID_DIAGNOSTICS_HPP_
#define MUJOCO_ROS2_CONTROL__VELOCITY_PID_DIAGNOSTICS_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include "control_toolbox/pid.hpp"

namespace mujoco_ros2_control
{
constexpr std::uint32_t kVelocityPidDiagnosticSchemaVersion = 2;

struct VelocityPidDiagnostic
{
  bool valid{false};
  std::string joint_name;
  double velocity_target{0.0};
  double velocity_actual{0.0};
  double velocity_error{0.0};
  double p_term{0.0};
  double i_term{0.0};
  double candidate_i_term{0.0};
  double peak_abs_candidate_i_term{0.0};
  double d_term{0.0};
  double pid_raw_output{0.0};
  double effort_clamped_output{0.0};
  bool i_clamp_active{false};
  bool effort_clamp_active{false};
  bool i_clamp_ever{false};
  bool effort_clamp_ever{false};
  std::uint64_t control_update_count{0};
  double kp{0.0};
  double ki{0.0};
  double kd{0.0};
  double i_max{0.0};
  double i_min{0.0};
  bool antiwindup{false};
  double effort_limit{0.0};
};

inline double diagnostic_tolerance(double first, double second)
{
  return 64.0 * std::numeric_limits<double>::epsilon() *
         std::max(1.0, std::max(std::abs(first), std::abs(second)));
}

// This helper deliberately calls computeCommand exactly once. Everything else
// reads the PID state produced by that call or mirrors the pre-existing URDF
// effort clamp, so collecting diagnostics cannot add a second integration step.
inline VelocityPidDiagnostic compute_velocity_pid_diagnostic(
  control_toolbox::Pid & pid, const std::string & joint_name, double velocity_target,
  double velocity_actual, std::uint64_t period_ns, bool has_effort_limit,
  double max_effort)
{
  VelocityPidDiagnostic diagnostic;
  diagnostic.valid = true;
  diagnostic.joint_name = joint_name;
  diagnostic.velocity_target = velocity_target;
  diagnostic.velocity_actual = velocity_actual;
  diagnostic.velocity_error = velocity_target - velocity_actual;

  const auto gains = pid.getGains();
  diagnostic.kp = gains.p_gain_;
  diagnostic.ki = gains.i_gain_;
  diagnostic.kd = gains.d_gain_;
  diagnostic.i_max = gains.i_max_;
  diagnostic.i_min = gains.i_min_;
  diagnostic.antiwindup = gains.antiwindup_;
  diagnostic.effort_limit = has_effort_limit ? max_effort : 0.0;

  double ignored_p_error = 0.0;
  double previous_i_error = 0.0;
  double ignored_d_error = 0.0;
  pid.getCurrentPIDErrors(ignored_p_error, previous_i_error, ignored_d_error);

  diagnostic.pid_raw_output = pid.computeCommand(diagnostic.velocity_error, period_ns);

  double p_error = 0.0;
  double i_error = 0.0;
  double d_error = 0.0;
  pid.getCurrentPIDErrors(p_error, i_error, d_error);
  diagnostic.p_term = diagnostic.kp * p_error;
  diagnostic.d_term = diagnostic.kd * d_error;
  // control_toolbox returns the exact P+I+D command. Taking the residual keeps
  // the published decomposition numerically identical to the command even when
  // anti-windup rewrites its internal integral error at a boundary.
  diagnostic.i_term =
    diagnostic.pid_raw_output - diagnostic.p_term - diagnostic.d_term;

  const double dt_seconds = static_cast<double>(period_ns) * 1e-9;
  diagnostic.candidate_i_term =
    diagnostic.ki * (previous_i_error + diagnostic.velocity_error * dt_seconds);
  const double upper_tolerance =
    diagnostic_tolerance(diagnostic.candidate_i_term, diagnostic.i_max);
  const double lower_tolerance =
    diagnostic_tolerance(diagnostic.candidate_i_term, diagnostic.i_min);
  diagnostic.i_clamp_active =
    (diagnostic.candidate_i_term > diagnostic.i_max + upper_tolerance ||
    diagnostic.candidate_i_term < diagnostic.i_min - lower_tolerance);

  diagnostic.effort_clamped_output = diagnostic.pid_raw_output;
  if (has_effort_limit) {
    diagnostic.effort_clamped_output = std::max(
      -max_effort, std::min(diagnostic.pid_raw_output, max_effort));
  }
  diagnostic.effort_clamp_active =
    std::abs(diagnostic.pid_raw_output - diagnostic.effort_clamped_output) >
    diagnostic_tolerance(diagnostic.pid_raw_output, diagnostic.effort_clamped_output);
  return diagnostic;
}

// Accumulate evidence at the controller update site, before any ROS publisher
// can sample or drop a diagnostic frame. The lifetime is one MujocoSystem
// instance; formal FRF runs create a fresh instance for every frequency.
inline VelocityPidDiagnostic accumulate_velocity_pid_evidence(
  const VelocityPidDiagnostic & previous, VelocityPidDiagnostic current)
{
  current.control_update_count = previous.control_update_count + 1u;
  current.peak_abs_candidate_i_term = std::max(
    previous.peak_abs_candidate_i_term, std::abs(current.candidate_i_term));
  current.i_clamp_ever = previous.i_clamp_ever || current.i_clamp_active;
  current.effort_clamp_ever =
    previous.effort_clamp_ever || current.effort_clamp_active;
  return current;
}
}  // namespace mujoco_ros2_control

#endif  // MUJOCO_ROS2_CONTROL__VELOCITY_PID_DIAGNOSTICS_HPP_
