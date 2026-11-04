#include "physics_step.hpp"
#include "shutdown_signal.hpp"

#include <mujoco/mujoco.h>

#include <cmath>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

struct ModelDeleter
{
  void operator()(mjModel * model) const
  {
    mj_deleteModel(model);
  }
};

struct DataDeleter
{
  void operator()(mjData * data) const
  {
    mj_deleteData(data);
  }
};

void require(bool condition, const std::string & message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool near(double lhs, double rhs, double tolerance = 1e-12)
{
  return std::fabs(lhs - rhs) <= tolerance;
}

std::unique_ptr<mjModel, ModelDeleter> make_model()
{
  constexpr char kModelXml[] =
    R"(
    <mujoco model="physics_step_test">
      <option timestep="0.01" gravity="0 0 0"/>
      <worldbody>
        <body>
          <joint name="slide" type="slide" axis="1 0 0"/>
          <geom type="sphere" size="0.05" mass="1"/>
        </body>
      </worldbody>
    </mujoco>
  )";

  mjVFS vfs;
  mj_defaultVFS(&vfs);
  require(
    mj_addBufferVFS(&vfs, "physics_step_test.xml", kModelXml, std::strlen(kModelXml)) == 0,
    "failed to add in-memory MJCF to VFS");

  char error[1024] = {};
  std::unique_ptr<mjModel, ModelDeleter> model(
    mj_loadXML("physics_step_test.xml", &vfs, error, sizeof(error)));
  mj_deleteVFS(&vfs);
  require(model != nullptr, std::string("failed to compile test MJCF: ") + error);
  return model;
}

class RecordingPlugin final : public mujoco_sim_ros2::MujocoPhysicsPlugin
{
public:
  RecordingPlugin(std::string name, std::vector<std::string> & events)
  : name_(std::move(name)), events_(events)
  {
  }

  void Configure(
    rclcpp::Node::SharedPtr &, rclcpp::NodeOptions, mjModel *, mjData *) override
  {
  }

  void Reset(mjModel *, mjData *) override
  {
    events_.push_back("reset:" + name_);
  }

  void PreUpdate(mjModel *, mjData * data) override
  {
    events_.push_back("pre:" + name_);
    pre_time = data->time;
  }

  void Update(mjModel *, mjData * data) override
  {
    events_.push_back("update:" + name_);
    update_time = data->time;
  }

  void PostUpdate(mjModel *, mjData * data) override
  {
    events_.push_back("post:" + name_);
    post_time = data->time;
  }

  double pre_time = -1.0;
  double update_time = -1.0;
  double post_time = -1.0;

private:
  std::string name_;
  std::vector<std::string> & events_;
};

void check_without_plugins(mjModel * model)
{
  std::unique_ptr<mjData, DataDeleter> data(mj_makeData(model));
  require(data != nullptr, "failed to allocate mjData");

  const mujoco_sim_ros2::PhysicsPluginList plugins;
  mujoco_sim_ros2::StepPhysics(model, data.get(), plugins);

  require(near(data->time, model->opt.timestep), "plugin-free path must advance one step");
}

void check_plugin_lifecycle(mjModel * model)
{
  std::unique_ptr<mjData, DataDeleter> data(mj_makeData(model));
  require(data != nullptr, "failed to allocate mjData");

  std::vector<std::string> events;
  auto first = std::make_shared<RecordingPlugin>("first", events);
  auto second = std::make_shared<RecordingPlugin>("second", events);
  const mujoco_sim_ros2::PhysicsPluginList plugins{first, second};

  mujoco_sim_ros2::StepPhysics(model, data.get(), plugins);

  const std::vector<std::string> expected{
    "pre:first", "pre:second",
    "update:first", "update:second",
    "post:first", "post:second"};
  require(events == expected, "plugins must receive one ordered lifecycle per physics step");
  require(near(first->pre_time, 0.0), "PreUpdate must run before integration");
  require(near(first->update_time, 0.0), "Update must run between mj_step1 and mj_step2");
  require(
    near(first->post_time, model->opt.timestep),
    "PostUpdate must run after mj_step2 advances simulation time");
  require(near(data->time, model->opt.timestep), "plugin path must advance exactly one step");
}

void check_time_rewind_resets_plugins(mjModel * model)
{
  std::unique_ptr<mjData, DataDeleter> data(mj_makeData(model));
  require(data != nullptr, "failed to allocate mjData");

  std::vector<std::string> events;
  auto first = std::make_shared<RecordingPlugin>("first", events);
  auto second = std::make_shared<RecordingPlugin>("second", events);
  const mujoco_sim_ros2::PhysicsPluginList plugins{first, second};

  data->time = 2.0;
  require(
    !mujoco_sim_ros2::ResetPhysicsPluginsIfTimeRewound(
      model, data.get(), plugins, 2.0),
    "equal simulation timestamps must not reset plugins");
  require(events.empty(), "plugins must not reset without a time rewind");

  data->time = 0.25;
  require(
    mujoco_sim_ros2::ResetPhysicsPluginsIfTimeRewound(
      model, data.get(), plugins, 2.0),
    "a simulation-time rewind must be detected");
  const std::vector<std::string> expected{"reset:first", "reset:second"};
  require(events == expected, "time rewind must reset every plugin in load order");
}

void check_shutdown_signal_is_cooperative()
{
  mujoco_sim_ros2::ResetShutdownSignal();
  require(
    !mujoco_sim_ros2::ShutdownSignalRequested(),
    "shutdown signal must start cleared");
  mujoco_sim_ros2::RequestShutdownFromSignal(SIGINT);
  require(
    mujoco_sim_ros2::ShutdownSignalRequested(),
    "signal handler must record a cooperative shutdown request");
  mujoco_sim_ros2::ResetShutdownSignal();
}

void check_physics_loop_uses_unified_step()
{
  std::ifstream source_file(MUJOCO_SIM_MAIN_SOURCE);
  require(source_file.good(), "failed to open main.cc for the integration contract check");
  std::ostringstream source_buffer;
  source_buffer << source_file.rdbuf();
  const std::string source = source_buffer.str();

  const std::regex unified_step_call("mujoco_sim_ros2::StepPhysics\\s*\\(");
  const std::regex rewind_reset_call(
    "mujoco_sim_ros2::ResetPhysicsPluginsIfTimeRewound\\s*\\(");
  const std::regex direct_step_call("\\bmj_step[12]?\\s*\\(");
  const auto unified_calls = std::distance(
    std::sregex_iterator(source.begin(), source.end(), unified_step_call),
    std::sregex_iterator());
  const auto rewind_reset_calls = std::distance(
    std::sregex_iterator(source.begin(), source.end(), rewind_reset_call),
    std::sregex_iterator());

  require(
    unified_calls == 2,
    "normal and re-synchronization paths must both call the unified physics step");
  require(
    !std::regex_search(source, direct_step_call),
    "main physics loop must not bypass the unified plugin lifecycle");
  require(
    rewind_reset_calls == 1,
    "main physics loop must notify plugins once when simulation time rewinds");
}

void check_shutdown_lifecycle_contract()
{
  std::ifstream source_file(MUJOCO_SIM_MAIN_SOURCE);
  require(source_file.good(), "failed to open main.cc for the shutdown contract check");
  std::ostringstream source_buffer;
  source_buffer << source_file.rdbuf();
  const std::string source = source_buffer.str();

  require(
    source.find("rclcpp::SignalHandlerOptions::None") != std::string::npos,
    "ROS must not install a competing signal handler");
  require(
    source.find("std::signal(SIGINT, mujoco_sim_ros2::RequestShutdownFromSignal)") !=
    std::string::npos,
    "SIGINT must only request cooperative shutdown");
  require(
    source.find("std::signal(SIGTERM, mujoco_sim_ros2::RequestShutdownFromSignal)") !=
    std::string::npos,
    "SIGTERM must only request cooperative shutdown");

  const auto clear_plugins = source.find("physics_plugins.clear();");
  const auto delete_data = source.find("mj_deleteData(d);", clear_plugins);
  const auto delete_model = source.find("mj_deleteModel(m);", delete_data);
  require(
    clear_plugins != std::string::npos && delete_data != std::string::npos &&
    delete_model != std::string::npos && clear_plugins < delete_data &&
    delete_data < delete_model,
    "plugin instances must stop before MuJoCo data and model are deleted");

  const auto join_physics = source.find("physicsthreadhandle.join();");
  const auto destroy_node = source.find("node.reset();", join_physics);
  const auto unload_plugins = source.find("physics_plugin_loader.reset();", join_physics);
  const auto shutdown_ros = source.find("rclcpp::shutdown();", join_physics);
  require(
    join_physics != std::string::npos && destroy_node != std::string::npos &&
    unload_plugins != std::string::npos && shutdown_ros != std::string::npos &&
    join_physics < destroy_node && destroy_node < unload_plugins &&
    unload_plugins < shutdown_ros,
    "the node must release plugin-created callback controls before plugin libraries unload");
}

}  // namespace

int main()
{
  try {
    auto model = make_model();
    check_without_plugins(model.get());
    check_plugin_lifecycle(model.get());
    check_time_rewind_resets_plugins(model.get());
    check_shutdown_signal_is_cooperative();
    check_physics_loop_uses_unified_step();
    check_shutdown_lifecycle_contract();
    std::cout << "MuJoCo " << mj_versionString()
              << " unified physics-step checks passed" << std::endl;
    return 0;
  } catch (const std::exception & error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
