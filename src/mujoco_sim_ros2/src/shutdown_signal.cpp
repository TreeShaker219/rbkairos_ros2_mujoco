#include "shutdown_signal.hpp"

#include <csignal>

namespace mujoco_sim_ros2
{
namespace
{

volatile std::sig_atomic_t shutdown_signal = 0;

}  // namespace

void RequestShutdownFromSignal(int signal_number) noexcept
{
  shutdown_signal = signal_number;
}

bool ShutdownSignalRequested() noexcept
{
  return shutdown_signal != 0;
}

void ResetShutdownSignal() noexcept
{
  shutdown_signal = 0;
}

}  // namespace mujoco_sim_ros2
