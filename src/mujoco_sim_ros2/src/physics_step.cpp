#include "physics_step.hpp"

namespace mujoco_sim_ros2
{

void StepPhysics(
  mjModel * model, mjData * data, const PhysicsPluginList & plugins)
{
  if (plugins.empty()) {
    mj_step(model, data);
    return;
  }

  for (const auto & plugin : plugins) {
    plugin->PreUpdate(model, data);
  }
  mj_step1(model, data);
  for (const auto & plugin : plugins) {
    plugin->Update(model, data);
  }
  mj_step2(model, data);
  for (const auto & plugin : plugins) {
    plugin->PostUpdate(model, data);
  }
}

bool ResetPhysicsPluginsIfTimeRewound(
  mjModel * model, mjData * data, const PhysicsPluginList & plugins,
  mjtNum previous_sim_time)
{
  if (data->time >= previous_sim_time) {
    return false;
  }

  for (const auto & plugin : plugins) {
    plugin->Reset(model, data);
  }
  return true;
}

}  // namespace mujoco_sim_ros2
