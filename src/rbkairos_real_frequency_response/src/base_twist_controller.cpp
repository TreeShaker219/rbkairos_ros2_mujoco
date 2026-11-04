#include "rbkairos_real_frequency_response/base_twist_controller.hpp"

#include <cmath>
#include <limits>
#include <utility>

#include <pluginlib/class_list_macros.hpp>

namespace rbkairos_real_frequency_response
{

namespace
{
const std::vector<std::string> kDefaultCommandInterfaces = {
  "summit_xl_2/base_footprint/readwrite_vx",
  "summit_xl_2/base_footprint/readwrite_vy",
  "summit_xl_2/base_footprint/readwrite_wz",
};

const std::vector<std::string> kDefaultStateInterfaces = {
  "summit_xl_2/base_footprint/read_x",
  "summit_xl_2/base_footprint/read_y",
  "summit_xl_2/base_footprint/read_qz",
  "summit_xl_2/base_footprint/read_qw",
  "summit_xl_2/base_footprint/readwrite_vx",
  "summit_xl_2/base_footprint/readwrite_vy",
  "summit_xl_2/base_footprint/readwrite_wz",
};
}  // namespace

BaseTwistController::BaseTwistController()
: command_interface_names_(kDefaultCommandInterfaces),
  state_interface_names_(kDefaultStateInterfaces),
  command_topic_("/base_cmd_vel"),
  odom_topic_("/rbkairos_base/odom"),
  emergency_stop_topic_("/emergency_stop"),
  command_timeout_s_(0.25),
  max_vx_abs_(0.05),
  max_vy_abs_(0.03),
  max_wz_abs_(0.10),
  publish_odom_(true),
  odom_publish_rate_(50.0),
  odom_frame_id_("world"),
  base_frame_id_("summit_xl_2/base_footprint"),
  stop_on_emergency_stop_(true),
  last_odom_publish_time_(0, 0, RCL_ROS_TIME)
{
  command_buffer_.writeFromNonRT(Command{});
  emergency_stop_buffer_.writeFromNonRT(false);
}

template<typename T>
void BaseTwistController::declare_if_not_declared(
  const std::string & name,
  const T & default_value)
{
  if (!get_node()->has_parameter(name)) {
    get_node()->declare_parameter<T>(name, default_value);
  }
}

controller_interface::CallbackReturn BaseTwistController::on_init()
{
  try {
    declare_if_not_declared("command_topic", command_topic_);
    declare_if_not_declared("odom_topic", odom_topic_);
    declare_if_not_declared("emergency_stop_topic", emergency_stop_topic_);
    declare_if_not_declared("command_timeout_s", command_timeout_s_);
    declare_if_not_declared("max_vx_abs", max_vx_abs_);
    declare_if_not_declared("max_vy_abs", max_vy_abs_);
    declare_if_not_declared("max_wz_abs", max_wz_abs_);
    declare_if_not_declared("publish_odom", publish_odom_);
    declare_if_not_declared("odom_publish_rate", odom_publish_rate_);
    declare_if_not_declared("odom_frame_id", odom_frame_id_);
    declare_if_not_declared("base_frame_id", base_frame_id_);
    declare_if_not_declared("stop_on_emergency_stop", stop_on_emergency_stop_);
    declare_if_not_declared("command_interfaces", command_interface_names_);
    declare_if_not_declared("state_interfaces", state_interface_names_);
    refresh_parameters();
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "BaseTwistController init failed: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

void BaseTwistController::refresh_parameters()
{
  command_topic_ = get_node()->get_parameter("command_topic").as_string();
  odom_topic_ = get_node()->get_parameter("odom_topic").as_string();
  emergency_stop_topic_ = get_node()->get_parameter("emergency_stop_topic").as_string();
  command_timeout_s_ = get_node()->get_parameter("command_timeout_s").as_double();
  max_vx_abs_ = get_node()->get_parameter("max_vx_abs").as_double();
  max_vy_abs_ = get_node()->get_parameter("max_vy_abs").as_double();
  max_wz_abs_ = get_node()->get_parameter("max_wz_abs").as_double();
  publish_odom_ = get_node()->get_parameter("publish_odom").as_bool();
  odom_publish_rate_ = get_node()->get_parameter("odom_publish_rate").as_double();
  odom_frame_id_ = get_node()->get_parameter("odom_frame_id").as_string();
  base_frame_id_ = get_node()->get_parameter("base_frame_id").as_string();
  stop_on_emergency_stop_ = get_node()->get_parameter("stop_on_emergency_stop").as_bool();
  command_interface_names_ = get_node()->get_parameter("command_interfaces").as_string_array();
  state_interface_names_ = get_node()->get_parameter("state_interfaces").as_string_array();
}

controller_interface::CallbackReturn BaseTwistController::on_configure(
  const rclcpp_lifecycle::State &)
{
  refresh_parameters();

  if (command_interface_names_.size() != 3) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Expected exactly 3 command interfaces, got %zu",
      command_interface_names_.size());
    return controller_interface::CallbackReturn::FAILURE;
  }
  if (state_interface_names_.size() != 7) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Expected exactly 7 state interfaces, got %zu",
      state_interface_names_.size());
    return controller_interface::CallbackReturn::FAILURE;
  }
  if (command_timeout_s_ <= 0.0 || odom_publish_rate_ <= 0.0) {
    RCLCPP_ERROR(get_node()->get_logger(), "Timeouts and publish rates must be positive");
    return controller_interface::CallbackReturn::FAILURE;
  }

  command_sub_ = get_node()->create_subscription<geometry_msgs::msg::Twist>(
    command_topic_,
    rclcpp::SystemDefaultsQoS(),
    [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
      Command command;
      command.vx = msg->linear.x;
      command.vy = msg->linear.y;
      command.wz = msg->angular.z;
      command.stamp = get_node()->get_clock()->now();
      command.valid = true;
      command_buffer_.writeFromNonRT(command);
    });

  emergency_stop_sub_ = get_node()->create_subscription<std_msgs::msg::Bool>(
    emergency_stop_topic_,
    rclcpp::SystemDefaultsQoS(),
    [this](const std_msgs::msg::Bool::SharedPtr msg) {
      emergency_stop_buffer_.writeFromNonRT(msg->data);
    });

  if (publish_odom_) {
    odom_pub_ = std::make_shared<realtime_tools::RealtimePublisher<nav_msgs::msg::Odometry>>(
      get_node()->create_publisher<nav_msgs::msg::Odometry>(odom_topic_, rclcpp::SystemDefaultsQoS()));
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
BaseTwistController::command_interface_configuration() const
{
  return {
    controller_interface::interface_configuration_type::INDIVIDUAL,
    command_interface_names_};
}

controller_interface::InterfaceConfiguration
BaseTwistController::state_interface_configuration() const
{
  return {
    controller_interface::interface_configuration_type::INDIVIDUAL,
    state_interface_names_};
}

controller_interface::CallbackReturn BaseTwistController::on_activate(
  const rclcpp_lifecycle::State &)
{
  command_buffer_.writeFromNonRT(Command{});
  emergency_stop_buffer_.writeFromNonRT(false);
  last_odom_publish_time_ = rclcpp::Time(0, 0, get_node()->get_clock()->get_clock_type());
  write_zero_commands();
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn BaseTwistController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  write_zero_commands();
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn BaseTwistController::on_shutdown(
  const rclcpp_lifecycle::State &)
{
  write_zero_commands();
  return controller_interface::CallbackReturn::SUCCESS;
}

double BaseTwistController::clamp(double value, double limit)
{
  return std::clamp(value, -std::abs(limit), std::abs(limit));
}

void BaseTwistController::write_commands(double vx, double vy, double wz)
{
  if (command_interfaces_.size() < 3) {
    return;
  }
  command_interfaces_[0].set_value(vx);
  command_interfaces_[1].set_value(vy);
  command_interfaces_[2].set_value(wz);
}

void BaseTwistController::write_zero_commands()
{
  write_commands(0.0, 0.0, 0.0);
}

void BaseTwistController::publish_odom(const rclcpp::Time & time)
{
  if (!publish_odom_ || !odom_pub_ || state_interfaces_.size() < 7) {
    return;
  }

  const double publish_period_s = 1.0 / odom_publish_rate_;
  if (last_odom_publish_time_.nanoseconds() != 0 &&
    (time - last_odom_publish_time_).seconds() < publish_period_s)
  {
    return;
  }

  if (odom_pub_->trylock()) {
    auto & msg = odom_pub_->msg_;
    msg.header.stamp = time;
    msg.header.frame_id = odom_frame_id_;
    msg.child_frame_id = base_frame_id_;
    msg.pose.pose.position.x = state_interfaces_[0].get_value();
    msg.pose.pose.position.y = state_interfaces_[1].get_value();
    msg.pose.pose.position.z = 0.0;
    msg.pose.pose.orientation.x = 0.0;
    msg.pose.pose.orientation.y = 0.0;
    msg.pose.pose.orientation.z = state_interfaces_[2].get_value();
    msg.pose.pose.orientation.w = state_interfaces_[3].get_value();
    msg.twist.twist.linear.x = state_interfaces_[4].get_value();
    msg.twist.twist.linear.y = state_interfaces_[5].get_value();
    msg.twist.twist.linear.z = 0.0;
    msg.twist.twist.angular.x = 0.0;
    msg.twist.twist.angular.y = 0.0;
    msg.twist.twist.angular.z = state_interfaces_[6].get_value();
    odom_pub_->unlockAndPublish();
    last_odom_publish_time_ = time;
  }
}

controller_interface::return_type BaseTwistController::update(
  const rclcpp::Time & time,
  const rclcpp::Duration &)
{
  double vx = 0.0;
  double vy = 0.0;
  double wz = 0.0;

  const Command * command = command_buffer_.readFromRT();
  const bool * emergency_stop = emergency_stop_buffer_.readFromRT();
  const bool stopped = stop_on_emergency_stop_ && emergency_stop && *emergency_stop;

  if (command && command->valid && !stopped) {
    const double age_s = (time - command->stamp).seconds();
    if (age_s >= 0.0 && age_s <= command_timeout_s_) {
      vx = clamp(command->vx, max_vx_abs_);
      vy = clamp(command->vy, max_vy_abs_);
      wz = clamp(command->wz, max_wz_abs_);
    }
  }

  write_commands(vx, vy, wz);
  publish_odom(time);

  return controller_interface::return_type::OK;
}

}  // namespace rbkairos_real_frequency_response

PLUGINLIB_EXPORT_CLASS(
  rbkairos_real_frequency_response::BaseTwistController,
  controller_interface::ControllerInterface)
