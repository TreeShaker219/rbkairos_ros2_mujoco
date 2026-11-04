#ifndef MUJOCO_SIM_ROS2__PHYSICS_STEP_HPP_
#define MUJOCO_SIM_ROS2__PHYSICS_STEP_HPP_

#include <memory>
#include <vector>

#include <mujoco/mujoco.h>

#include "mujoco_sim_ros2/mujoco_physics_plugin.hpp"

namespace mujoco_sim_ros2
{

using PhysicsPluginList = std::vector<std::shared_ptr<MujocoPhysicsPlugin>>;

void StepPhysics(
  mjModel * model, mjData * data, const PhysicsPluginList & plugins);

bool ResetPhysicsPluginsIfTimeRewound(
  mjModel * model, mjData * data, const PhysicsPluginList & plugins,
  mjtNum previous_sim_time);

}  // namespace mujoco_sim_ros2

#endif  // MUJOCO_SIM_ROS2__PHYSICS_STEP_HPP_
