// Copyright (c) 2025 Sangtaek Lee
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "hardware_interface/component_parser.hpp"
#include "hardware_interface/resource_manager.hpp"
#include "hardware_interface/system_interface.hpp"

#include "mujoco_ros2_control/mujoco_ros2_control.hpp"
#include "std_msgs/msg/string.hpp"

namespace
{
// MuJoCo debug migration: publish qpos in nv/DOF order so it lines up with
// qvel, qacc, and every qfrc_* array in MujocoDofState.
double qpos_value_for_dof(const mjModel *model, const mjData *data, int dof_index)
{
  const int joint_id = model->dof_jntid[dof_index];
  const int dof_adr = model->jnt_dofadr[joint_id];
  const int local = dof_index - dof_adr;
  const int qpos_adr = model->jnt_qposadr[joint_id];

  if (model->jnt_type[joint_id] == mjJNT_HINGE || model->jnt_type[joint_id] == mjJNT_SLIDE)
  {
    return data->qpos[qpos_adr + local];
  }
  if (model->jnt_type[joint_id] == mjJNT_FREE)
  {
    if (local < 3)
    {
      return data->qpos[qpos_adr + local];
    }
    return data->qpos[qpos_adr + 3 + (local - 3)];
  }
  if (model->jnt_type[joint_id] == mjJNT_BALL)
  {
    return data->qpos[qpos_adr + 1 + local];
  }
  return 0.0;
}

// MuJoCo debug migration: convert MuJoCo's mjtNum buffers into ROS float64[]
// fields without changing the nv ordering.
void assign_mjt_num_array(std::vector<double> &target, const mjtNum *source, int size)
{
  target.resize(size);
  for (int i = 0; i < size; ++i)
  {
    target[i] = static_cast<double>(source[i]);
  }
}

// MuJoCo debug migration: efc_force lives in constraint space, not DOF space.
// mj_mulJacTVec projects one grouped efc_force vector back to generalized DOF forces.
void multiply_constraint_group_to_qfrc(
  const mjModel *model, const mjData *data, const std::vector<mjtNum> &efc_force_group,
  std::vector<double> &qfrc_out)
{
  std::vector<mjtNum> qfrc(model->nv, 0.0);
  if (model->nv > 0 && !efc_force_group.empty())
  {
    mj_mulJacTVec(model, data, qfrc.data(), efc_force_group.data());
  }
  qfrc_out.resize(model->nv);
  for (int i = 0; i < model->nv; ++i)
  {
    qfrc_out[i] = static_cast<double>(qfrc[i]);
  }
}
}  // namespace

namespace mujoco_ros2_control
{
MujocoRos2Control::MujocoRos2Control(
  rclcpp::Node::SharedPtr &node, rclcpp::NodeOptions cm_node_option, mjModel *mujoco_model, mjData *mujoco_data)
    : node_(node),
      cm_node_option_(cm_node_option),
      mj_model_(mujoco_model),
      mj_data_(mujoco_data),
      logger_(rclcpp::get_logger(node_->get_name() + std::string(".mujoco_ros2_control"))),
      control_period_(rclcpp::Duration(1, 0)),
      last_update_sim_time_ros_(0, 0, RCL_ROS_TIME)
{
}

MujocoRos2Control::~MujocoRos2Control()
{
  stop_cm_thread_.store(true);
  if (cm_executor_) {
    cm_executor_->cancel();
  }

  if (cm_thread_.joinable()) {
    cm_thread_.join();
  }
  if (cm_executor_ && controller_manager_) {
    cm_executor_->remove_node(controller_manager_);
  }
  if (cm_executor_ && node_) {
    cm_executor_->remove_node(node_);
  }
}

std::string MujocoRos2Control::get_robot_description()
{
  // Getting robot description from parameter first. If not set trying from topic
  std::string robot_description;

  auto node = std::make_shared<rclcpp::Node>(
    "robot_description_node",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  if (node->has_parameter("robot_description"))
  {
    robot_description = node->get_parameter("robot_description").as_string();
    return robot_description;
  }

  RCLCPP_WARN(
    logger_,
    "Failed to get robot_description from parameter. Will listen on the ~/robot_description "
    "topic...");

  auto robot_description_sub = node->create_subscription<std_msgs::msg::String>(
    "robot_description", rclcpp::QoS(1).transient_local(),
    [&](const std_msgs::msg::String::SharedPtr msg)
    {
      if (!msg->data.empty() && robot_description.empty()) robot_description = msg->data;
    });

  while (robot_description.empty() && rclcpp::ok())
  {
    rclcpp::spin_some(node);
    RCLCPP_INFO(node->get_logger(), "Waiting for robot description message");
    rclcpp::sleep_for(std::chrono::milliseconds(500));
  }

  return robot_description;
}

void MujocoRos2Control::init()
{
  clock_publisher_ = node_->create_publisher<rosgraph_msgs::msg::Clock>("/clock", 10);

  std::string urdf_string = this->get_robot_description();

  // setup actuators and mechanism control node.
  std::vector<hardware_interface::HardwareInfo> control_hardware_info;
  try
  {
    control_hardware_info = hardware_interface::parse_control_resources_from_urdf(urdf_string);
  }
  catch (const std::runtime_error &ex)
  {
    RCLCPP_ERROR_STREAM(logger_, "Error parsing URDF : " << ex.what());
    return;
  }

  try
  {
    robot_hw_sim_loader_.reset(new pluginlib::ClassLoader<MujocoSystemInterface>(
      "mujoco_ros2_control", "mujoco_ros2_control::MujocoSystemInterface"));
  }
  catch (pluginlib::LibraryLoadException &ex)
  {
    RCLCPP_ERROR_STREAM(logger_, "Failed to create hardware interface loader:  " << ex.what());
    return;
  }

  std::unique_ptr<hardware_interface::ResourceManager> resource_manager =
    std::make_unique<hardware_interface::ResourceManager>();

  try
  {
    resource_manager->load_urdf(urdf_string, false, false);
  }
  catch (...)
  {
    RCLCPP_ERROR(logger_, "Error while initializing URDF!");
  }

  mujoco_system_interfaces_.clear();
  for (const auto &hardware : control_hardware_info)
  {
    std::string robot_hw_sim_type_str_ = hardware.hardware_class_type;
    std::unique_ptr<MujocoSystemInterface> mujoco_system;
    try
    {
      mujoco_system = std::unique_ptr<MujocoSystemInterface>(
        robot_hw_sim_loader_->createUnmanagedInstance(robot_hw_sim_type_str_));
    }
    catch (pluginlib::PluginlibException &ex)
    {
      RCLCPP_ERROR_STREAM(logger_, "The plugin failed to load. Error: " << ex.what());
      continue;
    }

    urdf::Model urdf_model;
    urdf_model.initString(urdf_string);
    if (!mujoco_system->init_sim(mj_model_, mj_data_, urdf_model, hardware))
    {
      RCLCPP_FATAL(logger_, "Could not initialize robot simulation interface");
      return;
    }

    auto *mujoco_system_interface = mujoco_system.get();
    resource_manager->import_component(std::move(mujoco_system), hardware);
    mujoco_system_interfaces_.push_back(mujoco_system_interface);

    rclcpp_lifecycle::State state(
      lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE,
      hardware_interface::lifecycle_state_names::ACTIVE);
    resource_manager->set_component_state(hardware.name, state);
  }

  // Create the controller manager
  RCLCPP_INFO(logger_, "Loading controller_manager");
  cm_executor_ = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
  controller_manager_.reset(new controller_manager::ControllerManager(
      std::move(resource_manager), cm_executor_,
      "controller_manager", node_->get_namespace(), cm_node_option_));

  cm_executor_->add_node(node_);
  cm_executor_->add_node(controller_manager_);

  if (!controller_manager_->has_parameter("update_rate"))
  {
    RCLCPP_ERROR_STREAM(logger_, "controller manager doesn't have an update_rate parameter");
    return;
  }

  auto update_rate = controller_manager_->get_parameter("update_rate").as_int();
  control_period_ = rclcpp::Duration(std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::duration<double>(1.0 / static_cast<double>(update_rate))));

  // Force setting of use_sime_time parameter
  controller_manager_->set_parameter(
    rclcpp::Parameter("use_sim_time", rclcpp::ParameterValue(true)));

  stop_cm_thread_.store(false);
  auto spin = [this]()
  {
    while (rclcpp::ok() && !stop_cm_thread_.load())
    {
      cm_executor_->spin_once();
    }
  };
  cm_thread_ = std::thread(spin);

  init_dof_names();
  // MuJoCo debug migration: replace the temporary JointTrajectory debug topics
  // with semantic messages for DOF-level solver data and per-contact forces.
  mujoco_dof_state_publisher_ =
    node_->create_publisher<mujoco_ros2_control_msgs::msg::MujocoDofState>(
      "mujoco_dof_state", 10);
  mujoco_contact_state_publisher_ =
    node_->create_publisher<mujoco_ros2_control_msgs::msg::MujocoContactState>(
      "mujoco_contact_state", 10);
}

void MujocoRos2Control::pre_update() {
  // Get the simulation time and period
  std::chrono::duration<double> sim_time(static_cast<double>(mj_data_->time));

  rclcpp::Time sim_time_ros(std::chrono::duration_cast<std::chrono::nanoseconds>(sim_time).count(), RCL_ROS_TIME);
  sim_time_ros_ = sim_time_ros;
  sim_period_ = sim_time_ros - last_update_sim_time_ros_;

  publish_sim_time(sim_time_ros);
}

void MujocoRos2Control::reset()
{
  std::chrono::duration<double> sim_time(static_cast<double>(mj_data_->time));
  sim_time_ros_ = rclcpp::Time(
    std::chrono::duration_cast<std::chrono::nanoseconds>(sim_time).count(), RCL_ROS_TIME);
  last_update_sim_time_ros_ = sim_time_ros_;
  sim_period_ = rclcpp::Duration(0, 0);
}

void MujocoRos2Control::update()
{
  if (sim_period_ >= control_period_) {
    controller_manager_->read(sim_time_ros_, sim_period_);
    controller_manager_->update(sim_time_ros_, sim_period_);
    last_update_sim_time_ros_ = sim_time_ros_;
  }
  // use same time as for read and update call - this is how it is done in ros2_control_node
  controller_manager_->write(sim_time_ros_, sim_period_);
}

void MujocoRos2Control::update_with_step() {
  mj_step1(mj_model_, mj_data_);

  if (sim_period_ >= control_period_) {
    controller_manager_->read(sim_time_ros_, sim_period_);
    controller_manager_->update(sim_time_ros_, sim_period_);
    last_update_sim_time_ros_ = sim_time_ros_;
  }

  // use same time as for read and update call - this is how it is done in ros2_control_node
  controller_manager_->write(sim_time_ros_, sim_period_);

  mj_step2(mj_model_, mj_data_);
}

void MujocoRos2Control::post_update()
{
  // MuJoCo debug migration: publish after mj_step2 so qpos/qvel/qacc and solver
  // diagnostics share one simulation timestamp.
  publish_mujoco_dof_state();
  publish_mujoco_contact_state();
}

void MujocoRos2Control::publish_sim_time(rclcpp::Time sim_time)
{
  // TODO(sangteak601)
  rosgraph_msgs::msg::Clock sim_time_msg;
  sim_time_msg.clock = sim_time;
  clock_publisher_->publish(sim_time_msg);
}

void MujocoRos2Control::init_dof_names()
{
  // MuJoCo debug migration: this name vector is the index map for every nv-length
  // debug array; dof_names_[i] must describe qvel[i], qacc[i], and qfrc_*[i].
  dof_names_.clear();
  dof_names_.reserve(mj_model_->nv);
  for (int j = 0; j < mj_model_->njnt; ++j)
  {
    const char *name_c = mj_id2name(mj_model_, mjOBJ_JOINT, j);
    std::string base_name = name_c ? std::string(name_c) : ("joint_" + std::to_string(j));
    int ndof = 1;
    if (mj_model_->jnt_type[j] == mjJNT_FREE)
    {
      ndof = 6;
    }
    else if (mj_model_->jnt_type[j] == mjJNT_BALL)
    {
      ndof = 3;
    }
    for (int k = 0; k < ndof; ++k)
    {
      std::string name = base_name;
      if (ndof > 1)
      {
        name += "#" + std::to_string(k);
      }
      dof_names_.push_back(name);
      if (static_cast<int>(dof_names_.size()) >= mj_model_->nv)
      {
        break;
      }
    }
    if (static_cast<int>(dof_names_.size()) >= mj_model_->nv)
    {
      break;
    }
  }
  while (static_cast<int>(dof_names_.size()) < mj_model_->nv)
  {
    int idx = static_cast<int>(dof_names_.size());
    dof_names_.push_back("dof_" + std::to_string(idx));
  }
}

void MujocoRos2Control::publish_mujoco_dof_state()
{
  // MuJoCo debug migration: /mujoco_dof_state is the semantic replacement for
  // the old /joint_states_full debug transport.
  if (!mujoco_dof_state_publisher_ || mujoco_dof_state_publisher_->get_subscription_count() == 0)
  {
    return;
  }

  mujoco_ros2_control_msgs::msg::MujocoDofState msg;
  msg.header.stamp = sim_time_ros_;
  msg.dof_names = dof_names_;
  msg.nefc = mj_data_->nefc;
  msg.ncon = mj_data_->ncon;

  // qpos is nq in MuJoCo, but the debug message is DOF-oriented. Map each DOF
  // to a representative position value so qpos[i] lines up with qvel[i].
  msg.qpos.resize(mj_model_->nv);
  for (int i = 0; i < mj_model_->nv; ++i)
  {
    msg.qpos[i] = qpos_value_for_dof(mj_model_, mj_data_, i);
  }

  // Direct nv-length MuJoCo force/state arrays keep their native DOF order.
  assign_mjt_num_array(msg.qvel, mj_data_->qvel, mj_model_->nv);
  assign_mjt_num_array(msg.qacc, mj_data_->qacc, mj_model_->nv);
  assign_mjt_num_array(msg.qfrc_applied, mj_data_->qfrc_applied, mj_model_->nv);
  assign_mjt_num_array(msg.qfrc_actuator, mj_data_->qfrc_actuator, mj_model_->nv);
  assign_mjt_num_array(msg.qfrc_passive, mj_data_->qfrc_passive, mj_model_->nv);
  assign_mjt_num_array(msg.qfrc_bias, mj_data_->qfrc_bias, mj_model_->nv);
  assign_mjt_num_array(msg.qfrc_constraint, mj_data_->qfrc_constraint, mj_model_->nv);

  msg.velocity_pid_diagnostic_schema_version = kVelocityPidDiagnosticSchemaVersion;
  for (const auto *system : mujoco_system_interfaces_)
  {
    for (const auto &diagnostic : system->get_velocity_pid_diagnostics())
    {
      if (!diagnostic.valid)
      {
        continue;
      }
      msg.velocity_pid_joint_names.push_back(diagnostic.joint_name);
      msg.velocity_pid_target.push_back(diagnostic.velocity_target);
      msg.velocity_pid_actual.push_back(diagnostic.velocity_actual);
      msg.velocity_pid_error.push_back(diagnostic.velocity_error);
      msg.velocity_pid_p_term.push_back(diagnostic.p_term);
      msg.velocity_pid_i_term.push_back(diagnostic.i_term);
      msg.velocity_pid_candidate_i_term.push_back(diagnostic.candidate_i_term);
      msg.velocity_pid_peak_abs_candidate_i_term.push_back(
        diagnostic.peak_abs_candidate_i_term);
      msg.velocity_pid_d_term.push_back(diagnostic.d_term);
      msg.velocity_pid_raw_output.push_back(diagnostic.pid_raw_output);
      msg.velocity_pid_effort_clamped_output.push_back(
        diagnostic.effort_clamped_output);
      msg.velocity_pid_i_clamp_active.push_back(diagnostic.i_clamp_active);
      msg.velocity_pid_effort_clamp_active.push_back(diagnostic.effort_clamp_active);
      msg.velocity_pid_i_clamp_ever.push_back(diagnostic.i_clamp_ever);
      msg.velocity_pid_effort_clamp_ever.push_back(diagnostic.effort_clamp_ever);
      msg.velocity_pid_control_update_count.push_back(diagnostic.control_update_count);
      msg.velocity_pid_kp.push_back(diagnostic.kp);
      msg.velocity_pid_ki.push_back(diagnostic.ki);
      msg.velocity_pid_kd.push_back(diagnostic.kd);
      msg.velocity_pid_i_max.push_back(diagnostic.i_max);
      msg.velocity_pid_i_min.push_back(diagnostic.i_min);
      msg.velocity_pid_antiwindup.push_back(diagnostic.antiwindup);
      msg.velocity_pid_effort_limit.push_back(diagnostic.effort_limit);
    }
  }

  // Split the solver constraint forces by efc_type first. Each temporary vector
  // stays length nefc so mj_mulJacTVec can project it back to nv generalized force.
  std::vector<mjtNum> equality(mj_data_->nefc, 0.0);
  std::vector<mjtNum> friction(mj_data_->nefc, 0.0);
  std::vector<mjtNum> limit(mj_data_->nefc, 0.0);
  std::vector<mjtNum> contact(mj_data_->nefc, 0.0);
  std::vector<mjtNum> other(mj_data_->nefc, 0.0);

  for (int i = 0; i < mj_data_->nefc; ++i)
  {
    switch (mj_data_->efc_type[i])
    {
      case mjCNSTR_EQUALITY:
        equality[i] = mj_data_->efc_force[i];
        break;
      case mjCNSTR_FRICTION_DOF:
      case mjCNSTR_FRICTION_TENDON:
        friction[i] = mj_data_->efc_force[i];
        break;
      case mjCNSTR_LIMIT_JOINT:
      case mjCNSTR_LIMIT_TENDON:
        limit[i] = mj_data_->efc_force[i];
        break;
      case mjCNSTR_CONTACT_FRICTIONLESS:
      case mjCNSTR_CONTACT_PYRAMIDAL:
      case mjCNSTR_CONTACT_ELLIPTIC:
        contact[i] = mj_data_->efc_force[i];
        break;
      default:
        other[i] = mj_data_->efc_force[i];
        break;
    }
  }

  // These outputs answer "which constraint family caused qfrc_constraint on
  // this DOF?" when inspecting qacc spikes in the CSV logs.
  multiply_constraint_group_to_qfrc(
    mj_model_, mj_data_, equality, msg.qfrc_constraint_equality);
  multiply_constraint_group_to_qfrc(
    mj_model_, mj_data_, friction, msg.qfrc_constraint_friction);
  multiply_constraint_group_to_qfrc(
    mj_model_, mj_data_, limit, msg.qfrc_constraint_limit);
  multiply_constraint_group_to_qfrc(
    mj_model_, mj_data_, contact, msg.qfrc_constraint_contact);
  multiply_constraint_group_to_qfrc(
    mj_model_, mj_data_, other, msg.qfrc_constraint_other);

  mujoco_dof_state_publisher_->publish(msg);
}

void MujocoRos2Control::publish_mujoco_contact_state()
{
  // MuJoCo debug migration: /mujoco_contact_state records the contact-side
  // evidence that complements the DOF-level qfrc_constraint_contact array.
  if (
    !mujoco_contact_state_publisher_ ||
    mujoco_contact_state_publisher_->get_subscription_count() == 0)
  {
    return;
  }

  mujoco_ros2_control_msgs::msg::MujocoContactState msg;
  msg.header.stamp = sim_time_ros_;
  msg.ncon = mj_data_->ncon;

  // Flatten per-contact fixed-size MuJoCo arrays into ROS arrays. Contact i uses
  // slices [3*i:3*i+3], [9*i:9*i+9], [5*i:5*i+5], and [6*i:6*i+6].
  msg.contact_id.reserve(mj_data_->ncon);
  msg.geom1_id.reserve(mj_data_->ncon);
  msg.geom2_id.reserve(mj_data_->ncon);
  msg.geom1_name.reserve(mj_data_->ncon);
  msg.geom2_name.reserve(mj_data_->ncon);
  msg.dim.reserve(mj_data_->ncon);
  msg.efc_address.reserve(mj_data_->ncon);
  msg.dist.reserve(mj_data_->ncon);
  msg.position_xyz.reserve(3 * mj_data_->ncon);
  msg.frame_3x3.reserve(9 * mj_data_->ncon);
  msg.friction.reserve(5 * mj_data_->ncon);
  msg.force_local_6d.reserve(6 * mj_data_->ncon);

  for (int i = 0; i < mj_data_->ncon; ++i)
  {
    const mjContact &contact = mj_data_->contact[i];
    const int geom1 = contact.geom[0];
    const int geom2 = contact.geom[1];
    const char *geom1_name = geom1 >= 0 ? mj_id2name(mj_model_, mjOBJ_GEOM, geom1) : nullptr;
    const char *geom2_name = geom2 >= 0 ? mj_id2name(mj_model_, mjOBJ_GEOM, geom2) : nullptr;

    msg.contact_id.push_back(i);
    msg.geom1_id.push_back(geom1);
    msg.geom2_id.push_back(geom2);
    msg.geom1_name.push_back(geom1_name ? std::string(geom1_name) : std::string());
    msg.geom2_name.push_back(geom2_name ? std::string(geom2_name) : std::string());
    msg.dim.push_back(contact.dim);
    msg.efc_address.push_back(contact.efc_address);
    msg.dist.push_back(static_cast<double>(contact.dist));

    for (int k = 0; k < 3; ++k)
    {
      msg.position_xyz.push_back(static_cast<double>(contact.pos[k]));
    }
    for (int k = 0; k < 9; ++k)
    {
      msg.frame_3x3.push_back(static_cast<double>(contact.frame[k]));
    }
    for (int k = 0; k < 5; ++k)
    {
      msg.friction.push_back(static_cast<double>(contact.friction[k]));
    }

    // mj_contactForce returns local contact-frame force/torque for contact i.
    mjtNum force[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    mj_contactForce(mj_model_, mj_data_, i, force);
    for (int k = 0; k < 6; ++k)
    {
      msg.force_local_6d.push_back(static_cast<double>(force[k]));
    }
  }

  mujoco_contact_state_publisher_->publish(msg);
}

}  // namespace mujoco_ros2_control
