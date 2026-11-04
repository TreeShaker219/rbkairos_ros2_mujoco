#include "rbkairos/base_ground_truth_plugin.hpp"

#include <chrono>
#include <cmath>
#include <limits>

#include "pluginlib/class_list_macros.hpp"

namespace rbkairos
{

namespace
{
constexpr double kNanosecondsPerSecond = 1000000000.0;
}

BaseGroundTruthPlugin::BaseGroundTruthPlugin()
: logger_(rclcpp::get_logger("base_ground_truth_plugin")),
  publish_pose_(true),
  publish_twist_(true),
  publish_odom_(true),
  print_enabled_(true),
  publish_rate_hz_(50.0),
  print_rate_hz_(5.0),
  last_publish_time_(-std::numeric_limits<double>::infinity()),
  last_print_time_(-std::numeric_limits<double>::infinity()),
  model_(nullptr),
  body_id_(-1),
  enabled_(false)
{
}

void BaseGroundTruthPlugin::Configure(
  rclcpp::Node::SharedPtr &node, rclcpp::NodeOptions cm_node_option, mjModel *model, mjData *data)
{
  (void)cm_node_option;
  (void)data;

  node_ = node;
  logger_ = node_->get_logger();

  body_name_ = declare_or_get_parameter<std::string>(
    "base_ground_truth.body_name", "summit_xl_2/base_footprint");
  world_frame_id_ =
    declare_or_get_parameter<std::string>("base_ground_truth.world_frame_id", "world");
  child_frame_id_ =
    declare_or_get_parameter<std::string>("base_ground_truth.child_frame_id", "");
  odom_frame_id_ = declare_or_get_parameter<std::string>("base_ground_truth.odom_frame_id", "");

  pose_topic_ =
    declare_or_get_parameter<std::string>("base_ground_truth.pose_topic", "base_ground_truth_pose");
  twist_topic_ = declare_or_get_parameter<std::string>(
    "base_ground_truth.twist_topic", "base_ground_truth_twist");
  odom_topic_ =
    declare_or_get_parameter<std::string>("base_ground_truth.odom_topic", "base_ground_truth_odom");

  publish_pose_ = declare_or_get_parameter<bool>("base_ground_truth.publish_pose", true);
  publish_twist_ = declare_or_get_parameter<bool>("base_ground_truth.publish_twist", true);
  publish_odom_ = declare_or_get_parameter<bool>("base_ground_truth.publish_odom", true);
  print_enabled_ = declare_or_get_parameter<bool>("base_ground_truth.print_enabled", true);

  publish_rate_hz_ = declare_or_get_parameter<double>("base_ground_truth.publish_rate_hz", 50.0);
  print_rate_hz_ = declare_or_get_parameter<double>("base_ground_truth.print_rate_hz", 5.0);

  if (child_frame_id_.empty())
  {
    child_frame_id_ = body_name_;
  }
  if (odom_frame_id_.empty())
  {
    odom_frame_id_ = world_frame_id_;
  }

  if (publish_pose_)
  {
    pose_publisher_ =
      node_->create_publisher<geometry_msgs::msg::PoseStamped>(pose_topic_, rclcpp::SensorDataQoS());
  }
  if (publish_twist_)
  {
    twist_publisher_ = node_->create_publisher<geometry_msgs::msg::TwistStamped>(
      twist_topic_, rclcpp::SensorDataQoS());
  }
  if (publish_odom_)
  {
    odom_publisher_ =
      node_->create_publisher<nav_msgs::msg::Odometry>(odom_topic_, rclcpp::SensorDataQoS());
  }

  enabled_ = resolve_body(model);
  if (enabled_)
  {
    RCLCPP_INFO(
      logger_,
      "Base ground-truth tracking enabled for body '%s' (id=%d), world_frame='%s', child_frame='%s'",
      body_name_.c_str(), body_id_, world_frame_id_.c_str(), child_frame_id_.c_str());
  }
}

void BaseGroundTruthPlugin::Reset(mjModel *model, mjData *data)
{
  (void)data;
  enabled_ = resolve_body(model);
  last_publish_time_ = -std::numeric_limits<double>::infinity();
  last_print_time_ = -std::numeric_limits<double>::infinity();
}

void BaseGroundTruthPlugin::Update(mjModel *model, mjData *data)
{
  if (model != model_)
  {
    enabled_ = resolve_body(model);
  }

  if (!enabled_ || model == nullptr || data == nullptr)
  {
    return;
  }

  const mjtNum *pos = data->xpos + 3 * body_id_;
  const mjtNum *quat = data->xquat + 4 * body_id_;

  mjtNum world_velocity[6] = {0, 0, 0, 0, 0, 0};
  mjtNum local_velocity[6] = {0, 0, 0, 0, 0, 0};
  mj_objectVelocity(model, data, mjOBJ_BODY, body_id_, world_velocity, 0);
  mj_objectVelocity(model, data, mjOBJ_BODY, body_id_, local_velocity, 1);

  if (should_publish(data->time))
  {
    publish_messages(
      sim_time_to_ros_time(data->time), pos, quat, world_velocity, local_velocity);
    last_publish_time_ = data->time;
  }

  if (should_print(data->time))
  {
    print_sample(pos, quat, world_velocity);
    last_print_time_ = data->time;
  }
}

bool BaseGroundTruthPlugin::resolve_body(mjModel *model)
{
  model_ = model;
  body_id_ = -1;

  if (model_ == nullptr)
  {
    RCLCPP_ERROR(logger_, "Cannot resolve ground-truth body '%s': mjModel is null", body_name_.c_str());
    return false;
  }

  body_id_ = mj_name2id(model_, mjOBJ_BODY, body_name_.c_str());
  if (body_id_ < 0)
  {
    RCLCPP_ERROR(
      logger_, "Cannot resolve ground-truth body '%s'. Check the exact MJCF body name.",
      body_name_.c_str());
    return false;
  }

  return true;
}

bool BaseGroundTruthPlugin::should_publish(double sim_time) const
{
  if (!publish_pose_ && !publish_twist_ && !publish_odom_)
  {
    return false;
  }
  if (publish_rate_hz_ <= 0.0)
  {
    return true;
  }
  return sim_time - last_publish_time_ >= 1.0 / publish_rate_hz_;
}

bool BaseGroundTruthPlugin::should_print(double sim_time) const
{
  if (!print_enabled_ || print_rate_hz_ <= 0.0)
  {
    return false;
  }
  return sim_time - last_print_time_ >= 1.0 / print_rate_hz_;
}

rclcpp::Time BaseGroundTruthPlugin::sim_time_to_ros_time(double sim_time) const
{
  const auto nanoseconds = static_cast<int64_t>(std::llround(sim_time * kNanosecondsPerSecond));
  return rclcpp::Time(nanoseconds, RCL_ROS_TIME);
}

double BaseGroundTruthPlugin::yaw_from_mujoco_quat(const mjtNum quat[4]) const
{
  const double w = quat[0];
  const double x = quat[1];
  const double y = quat[2];
  const double z = quat[3];
  return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

void BaseGroundTruthPlugin::publish_messages(
  const rclcpp::Time &stamp, const mjtNum pos[3], const mjtNum quat[4],
  const mjtNum world_velocity[6], const mjtNum local_velocity[6])
{
  if (pose_publisher_)
  {
    geometry_msgs::msg::PoseStamped msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = world_frame_id_;
    msg.pose.position.x = pos[0];
    msg.pose.position.y = pos[1];
    msg.pose.position.z = pos[2];
    msg.pose.orientation.w = quat[0];
    msg.pose.orientation.x = quat[1];
    msg.pose.orientation.y = quat[2];
    msg.pose.orientation.z = quat[3];
    pose_publisher_->publish(msg);
  }

  if (twist_publisher_)
  {
    geometry_msgs::msg::TwistStamped msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = world_frame_id_;
    msg.twist.angular.x = world_velocity[0];
    msg.twist.angular.y = world_velocity[1];
    msg.twist.angular.z = world_velocity[2];
    msg.twist.linear.x = world_velocity[3];
    msg.twist.linear.y = world_velocity[4];
    msg.twist.linear.z = world_velocity[5];
    twist_publisher_->publish(msg);
  }

  if (odom_publisher_)
  {
    nav_msgs::msg::Odometry msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = odom_frame_id_;
    msg.child_frame_id = child_frame_id_;
    msg.pose.pose.position.x = pos[0];
    msg.pose.pose.position.y = pos[1];
    msg.pose.pose.position.z = pos[2];
    msg.pose.pose.orientation.w = quat[0];
    msg.pose.pose.orientation.x = quat[1];
    msg.pose.pose.orientation.y = quat[2];
    msg.pose.pose.orientation.z = quat[3];
    msg.twist.twist.angular.x = local_velocity[0];
    msg.twist.twist.angular.y = local_velocity[1];
    msg.twist.twist.angular.z = local_velocity[2];
    msg.twist.twist.linear.x = local_velocity[3];
    msg.twist.twist.linear.y = local_velocity[4];
    msg.twist.twist.linear.z = local_velocity[5];
    odom_publisher_->publish(msg);
  }
}

void BaseGroundTruthPlugin::print_sample(
  const mjtNum pos[3], const mjtNum quat[4], const mjtNum world_velocity[6]) const
{
  RCLCPP_INFO(
    logger_, "[base_gt %s] x=%.3f y=%.3f z=%.3f yaw=%.3f\nvx=%.3f vy=%.3f wz=%.3f",
    body_name_.c_str(), pos[0], pos[1], pos[2], yaw_from_mujoco_quat(quat),
    world_velocity[3], world_velocity[4], world_velocity[2]);
}

}  // namespace rbkairos

PLUGINLIB_EXPORT_CLASS(rbkairos::BaseGroundTruthPlugin, mujoco_sim_ros2::MujocoPhysicsPlugin)
