#include "rbkairos/traction_tendon_observer_plugin.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>

#include "pluginlib/class_list_macros.hpp"

namespace rbkairos
{

namespace
{
constexpr double kNanosecondsPerSecond = 1000000000.0;
constexpr double kMinimumCableLength = 1e-12;

// Float64MultiArray schema v1. Indices are part of the recorder/analyzer contract.
constexpr std::array<const char *, TractionTendonObserverPlugin::kStateFieldCount>
kStateFieldNames{{
  "sim_time_s",
  "tendon_length_m",
  "tendon_length0_m",
  "lower_limit_m",
  "upper_limit_m",
  "extension_over_upper_limit_m",
  "tension_n",
  "limit_active_flag",
  "force_x_n",
  "force_y_n",
  "force_z_n",
  "attach_x_m",
  "attach_y_m",
  "attach_z_m",
  "anchor_x_m",
  "anchor_y_m",
  "anchor_z_m",
}};

std::string schema_label()
{
  std::string label = "traction_tendon_state_v1:";
  for (std::size_t index = 0; index < kStateFieldNames.size(); ++index) {
    if (index > 0) {
      label += ',';
    }
    label += kStateFieldNames[index];
  }
  return label;
}
}  // namespace

constexpr int TractionTendonObserverPlugin::kStateSchemaVersion;
constexpr std::size_t TractionTendonObserverPlugin::kStateFieldCount;

TractionTendonObserverPlugin::TractionTendonObserverPlugin()
: logger_(rclcpp::get_logger("traction_tendon_observer_plugin")),
  enabled_(false),
  print_enabled_(true),
  publish_rate_hz_(100.0),
  print_rate_hz_(5.0),
  last_publish_time_s_(-std::numeric_limits<double>::infinity()),
  last_print_time_s_(-std::numeric_limits<double>::infinity()),
  model_(nullptr),
  tendon_id_(-1),
  sensor_id_(-1),
  attach_site_id_(-1),
  anchor_site_id_(-1)
{
}

void TractionTendonObserverPlugin::Configure(
  rclcpp::Node::SharedPtr & node, rclcpp::NodeOptions cm_node_option, mjModel * model,
  mjData * data)
{
  (void)cm_node_option;
  (void)data;

  node_ = node;
  logger_ = node_->get_logger();

  tendon_name_ =
    declare_or_get_parameter<std::string>("traction_tendon.tendon_name", "traction_cable_x");
  sensor_name_ = declare_or_get_parameter<std::string>(
    "traction_tendon.sensor_name", "traction_cable_x_limit_force");
  attach_site_name_ = declare_or_get_parameter<std::string>(
    "traction_tendon.attach_site_name", "traction_attach_x");
  anchor_site_name_ = declare_or_get_parameter<std::string>(
    "traction_tendon.anchor_site_name", "traction_anchor_x");
  publish_rate_hz_ =
    declare_or_get_parameter<double>("traction_tendon.publish_rate_hz", 100.0);
  print_enabled_ = declare_or_get_parameter<bool>("traction_tendon.print_enabled", true);
  print_rate_hz_ = declare_or_get_parameter<double>("traction_tendon.print_rate_hz", 5.0);
  state_topic_ = declare_or_get_parameter<std::string>(
    "traction_tendon.state_topic", "/traction_tendon/state");
  wrench_topic_ = declare_or_get_parameter<std::string>(
    "traction_tendon.wrench_topic", "/traction_tendon/wrench");
  frame_id_ = declare_or_get_parameter<std::string>("traction_tendon.frame_id", "world");

  publish_rate_hz_ = std::max(0.0, publish_rate_hz_);
  print_rate_hz_ = std::max(0.0, print_rate_hz_);

  state_publisher_ =
    node_->create_publisher<std_msgs::msg::Float64MultiArray>(
    state_topic_,
    rclcpp::SensorDataQoS());
  wrench_publisher_ = node_->create_publisher<geometry_msgs::msg::WrenchStamped>(
    wrench_topic_, rclcpp::SensorDataQoS());

  reset_publish_schedule();
  enabled_ = resolve_model_objects(model);
  if (enabled_) {
    RCLCPP_INFO(
      logger_,
      "Traction tendon observer enabled: tendon='%s', sensor='%s', attach='%s', anchor='%s', "
      "schema_version=%d",
      tendon_name_.c_str(), sensor_name_.c_str(), attach_site_name_.c_str(),
      anchor_site_name_.c_str(), kStateSchemaVersion);
  }
}

void TractionTendonObserverPlugin::Reset(mjModel * model, mjData * data)
{
  (void)data;
  reset_publish_schedule();
  enabled_ = resolve_model_objects(model);
}

void TractionTendonObserverPlugin::PostUpdate(mjModel * model, mjData * data)
{
  if (model == nullptr || data == nullptr) {
    return;
  }
  if (model != model_) {
    reset_publish_schedule();
    enabled_ = resolve_model_objects(model);
  }
  if (!enabled_) {
    return;
  }

  TendonSample sample;
  if (!compute_sample(model, data, sample)) {
    return;
  }

  if (should_publish(sample.sim_time_s)) {
    publish_sample(sample);
    last_publish_time_s_ = sample.sim_time_s;
  }
  if (should_print(sample.sim_time_s)) {
    print_sample(sample);
    last_print_time_s_ = sample.sim_time_s;
  }
}

bool TractionTendonObserverPlugin::resolve_model_objects(const mjModel * model)
{
  model_ = model;
  tendon_id_ = -1;
  sensor_id_ = -1;
  attach_site_id_ = -1;
  anchor_site_id_ = -1;

  if (model_ == nullptr) {
    RCLCPP_ERROR(logger_, "Cannot resolve traction tendon objects: mjModel is null");
    return false;
  }

  tendon_id_ = mj_name2id(model_, mjOBJ_TENDON, tendon_name_.c_str());
  sensor_id_ = mj_name2id(model_, mjOBJ_SENSOR, sensor_name_.c_str());
  attach_site_id_ = mj_name2id(model_, mjOBJ_SITE, attach_site_name_.c_str());
  anchor_site_id_ = mj_name2id(model_, mjOBJ_SITE, anchor_site_name_.c_str());

  if (tendon_id_ < 0 || sensor_id_ < 0 || attach_site_id_ < 0 || anchor_site_id_ < 0) {
    RCLCPP_ERROR(
      logger_,
      "Cannot resolve traction tendon objects: tendon='%s'(%d), sensor='%s'(%d), attach='%s'(%d), "
      "anchor='%s'(%d)",
      tendon_name_.c_str(), tendon_id_, sensor_name_.c_str(), sensor_id_, attach_site_name_.c_str(),
      attach_site_id_, anchor_site_name_.c_str(), anchor_site_id_);
    return false;
  }
  if (!model_->tendon_limited[tendon_id_]) {
    RCLCPP_ERROR(logger_, "Tendon '%s' is not limit-enabled", tendon_name_.c_str());
    return false;
  }
  const int sensor_address = model_->sensor_adr[sensor_id_];
  if (
    model_->sensor_type[sensor_id_] != mjSENS_TENDONLIMITFRC ||
    model_->sensor_objtype[sensor_id_] != mjOBJ_TENDON ||
    model_->sensor_objid[sensor_id_] != tendon_id_ || model_->sensor_dim[sensor_id_] != 1 ||
    model_->sensor_needstage[sensor_id_] != mjSTAGE_ACC ||
    model_->sensor_datatype[sensor_id_] != mjDATATYPE_REAL || sensor_address < 0 ||
    sensor_address + model_->sensor_dim[sensor_id_] > model_->nsensordata ||
    model_->sensor_noise[sensor_id_] != 0.0 || model_->sensor_cutoff[sensor_id_] != 0.0)
  {
    RCLCPP_ERROR(
      logger_, "Sensor '%s' is not a scalar tendonlimitfrc sensor for tendon '%s'",
      sensor_name_.c_str(), tendon_name_.c_str());
    return false;
  }

  const double lower_limit = model_->tendon_range[2 * tendon_id_];
  const double upper_limit = model_->tendon_range[2 * tendon_id_ + 1];
  if (
    !std::isfinite(lower_limit) || !std::isfinite(upper_limit) || lower_limit != 0.0 ||
    upper_limit <= model_->tendon_length0[tendon_id_] ||
    model_->tendon_stiffness[tendon_id_] != 0.0 ||
    model_->tendon_damping[tendon_id_] != 0.0 || model_->tendon_frictionloss[tendon_id_] != 0.0)
  {
    RCLCPP_ERROR(
      logger_, "Tendon '%s' is not a passive upper-limit-only cable", tendon_name_.c_str());
    return false;
  }

  for (int actuator_id = 0; actuator_id < model_->nu; ++actuator_id) {
    if (
      model_->actuator_trntype[actuator_id] == mjTRN_TENDON &&
      model_->actuator_trnid[2 * actuator_id] == tendon_id_)
    {
      RCLCPP_ERROR(logger_, "Tendon '%s' must not have an actuator", tendon_name_.c_str());
      return false;
    }
  }

  const int path_address = model_->tendon_adr[tendon_id_];
  if (
    model_->tendon_num[tendon_id_] != 2 || path_address<0 || path_address + 2> model_->nwrap ||
    model_->wrap_type[path_address] != mjWRAP_SITE ||
    model_->wrap_objid[path_address] != anchor_site_id_ ||
    model_->wrap_type[path_address + 1] != mjWRAP_SITE ||
    model_->wrap_objid[path_address + 1] != attach_site_id_)
  {
    RCLCPP_ERROR(
      logger_, "Tendon '%s' is not the configured two-site anchor-to-attach cable",
      tendon_name_.c_str());
    return false;
  }
  return true;
}

bool TractionTendonObserverPlugin::compute_sample(
  const mjModel * model, const mjData * data, TendonSample & sample) const
{
  if (
    tendon_id_ < 0 || sensor_id_ < 0 || attach_site_id_ < 0 || anchor_site_id_ < 0 ||
    model->sensor_adr[sensor_id_] < 0)
  {
    return false;
  }

  sample.sim_time_s = data->time;
  sample.tendon_length_m = data->ten_length[tendon_id_];
  sample.tendon_length0_m = model->tendon_length0[tendon_id_];
  sample.lower_limit_m = model->tendon_range[2 * tendon_id_];
  sample.upper_limit_m = model->tendon_range[2 * tendon_id_ + 1];
  sample.extension_over_upper_limit_m =
    std::max(0.0, sample.tendon_length_m - sample.upper_limit_m);
  sample.limit_active = false;
  if (data->tendon_efcadr[tendon_id_] >= 0) {
    for (int constraint_id = 0; constraint_id < data->nefc; ++constraint_id) {
      if (
        data->efc_type[constraint_id] == mjCNSTR_LIMIT_TENDON &&
        data->efc_id[constraint_id] == tendon_id_)
      {
        sample.limit_active = true;
        break;
      }
    }
  }
  sample.tension_n = data->sensordata[model->sensor_adr[sensor_id_]];
  if (!std::isfinite(sample.tension_n) || sample.tension_n < 0.0) {
    return false;
  }

  const mjtNum * attach = data->site_xpos + 3 * attach_site_id_;
  const mjtNum * anchor = data->site_xpos + 3 * anchor_site_id_;
  for (std::size_t index = 0; index < 3; ++index) {
    sample.attach_pos_m[index] = attach[index];
    sample.anchor_pos_m[index] = anchor[index];
  }

  const std::array<double, 3> toward_anchor{{
    sample.anchor_pos_m[0] - sample.attach_pos_m[0],
    sample.anchor_pos_m[1] - sample.attach_pos_m[1],
    sample.anchor_pos_m[2] - sample.attach_pos_m[2],
  }};
  const double length = std::sqrt(
    toward_anchor[0] * toward_anchor[0] + toward_anchor[1] * toward_anchor[1] +
    toward_anchor[2] * toward_anchor[2]);
  if (length > kMinimumCableLength) {
    for (std::size_t index = 0; index < 3; ++index) {
      sample.force_n[index] = sample.tension_n * toward_anchor[index] / length;
    }
  }
  return true;
}

void TractionTendonObserverPlugin::reset_publish_schedule()
{
  last_publish_time_s_ = -std::numeric_limits<double>::infinity();
  last_print_time_s_ = -std::numeric_limits<double>::infinity();
}

bool TractionTendonObserverPlugin::should_publish(double sim_time_s) const
{
  if (!state_publisher_ || !wrench_publisher_) {
    return false;
  }
  if (publish_rate_hz_ <= 0.0) {
    return true;
  }
  return sim_time_s - last_publish_time_s_ >= 1.0 / publish_rate_hz_ - 1e-12;
}

bool TractionTendonObserverPlugin::should_print(double sim_time_s) const
{
  if (!print_enabled_) {
    return false;
  }
  if (print_rate_hz_ <= 0.0) {
    return true;
  }
  return sim_time_s - last_print_time_s_ >= 1.0 / print_rate_hz_ - 1e-12;
}

rclcpp::Time TractionTendonObserverPlugin::sim_time_to_ros_time(double sim_time_s) const
{
  const auto nanoseconds =
    static_cast<int64_t>(std::llround(sim_time_s * kNanosecondsPerSecond));
  return rclcpp::Time(nanoseconds, RCL_ROS_TIME);
}

void TractionTendonObserverPlugin::publish_sample(const TendonSample & sample)
{
  std_msgs::msg::Float64MultiArray state;
  state.layout.dim.resize(1);
  state.layout.dim[0].label = schema_label();
  state.layout.dim[0].size = kStateFieldCount;
  state.layout.dim[0].stride = kStateFieldCount;
  state.data = {
    sample.sim_time_s,
    sample.tendon_length_m,
    sample.tendon_length0_m,
    sample.lower_limit_m,
    sample.upper_limit_m,
    sample.extension_over_upper_limit_m,
    sample.tension_n,
    sample.limit_active ? 1.0 : 0.0,
    sample.force_n[0],
    sample.force_n[1],
    sample.force_n[2],
    sample.attach_pos_m[0],
    sample.attach_pos_m[1],
    sample.attach_pos_m[2],
    sample.anchor_pos_m[0],
    sample.anchor_pos_m[1],
    sample.anchor_pos_m[2],
  };
  state_publisher_->publish(state);

  geometry_msgs::msg::WrenchStamped wrench;
  wrench.header.stamp = sim_time_to_ros_time(sample.sim_time_s);
  wrench.header.frame_id = frame_id_;
  wrench.wrench.force.x = sample.force_n[0];
  wrench.wrench.force.y = sample.force_n[1];
  wrench.wrench.force.z = sample.force_n[2];
  wrench_publisher_->publish(wrench);
}

void TractionTendonObserverPlugin::print_sample(const TendonSample & sample) const
{
  RCLCPP_INFO(
    logger_,
    "traction tendon t=%.3f s length=%.6f m upper=%.6f m extension=%.6f m tension=%.3f N "
    "active=%s force=(%.3f, %.3f, %.3f) N",
    sample.sim_time_s, sample.tendon_length_m, sample.upper_limit_m,
    sample.extension_over_upper_limit_m, sample.tension_n, sample.limit_active ? "true" : "false",
    sample.force_n[0], sample.force_n[1], sample.force_n[2]);
}

}  // namespace rbkairos

PLUGINLIB_EXPORT_CLASS(
  rbkairos::TractionTendonObserverPlugin, mujoco_sim_ros2::MujocoPhysicsPlugin)
