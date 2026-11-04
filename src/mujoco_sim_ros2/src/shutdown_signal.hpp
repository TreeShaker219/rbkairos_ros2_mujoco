#ifndef MUJOCO_SIM_ROS2__SHUTDOWN_SIGNAL_HPP_
#define MUJOCO_SIM_ROS2__SHUTDOWN_SIGNAL_HPP_

namespace mujoco_sim_ros2
{

void RequestShutdownFromSignal(int signal_number) noexcept;
bool ShutdownSignalRequested() noexcept;
void ResetShutdownSignal() noexcept;

}  // namespace mujoco_sim_ros2

#endif  // MUJOCO_SIM_ROS2__SHUTDOWN_SIGNAL_HPP_
