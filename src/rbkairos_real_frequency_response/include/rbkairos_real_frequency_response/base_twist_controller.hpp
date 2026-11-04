#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <controller_interface/controller_interface.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <realtime_tools/realtime_buffer.hpp>
#include <realtime_tools/realtime_publisher.hpp>
#include <std_msgs/msg/bool.hpp>

namespace rbkairos_real_frequency_response
{

class BaseTwistController : public controller_interface::ControllerInterface
{
public:
  BaseTwistController();

  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_shutdown(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::return_type update(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

private:
  struct Command
  {
    double vx{0.0};
    double vy{0.0};
    double wz{0.0};
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
    bool valid{false};
  };

  template<typename T>
  void declare_if_not_declared(const std::string & name, const T & default_value);

  void refresh_parameters();
  void write_zero_commands();
  void write_commands(double vx, double vy, double wz);
  void publish_odom(const rclcpp::Time & time);
  static double clamp(double value, double limit);

  std::vector<std::string> command_interface_names_;
  std::vector<std::string> state_interface_names_;

  std::string command_topic_;
  std::string odom_topic_;
  std::string emergency_stop_topic_;
  double command_timeout_s_;
  double max_vx_abs_;
  double max_vy_abs_;
  double max_wz_abs_;
  bool publish_odom_;
  double odom_publish_rate_;
  std::string odom_frame_id_;
  std::string base_frame_id_;
  bool stop_on_emergency_stop_;

  realtime_tools::RealtimeBuffer<Command> command_buffer_;
  realtime_tools::RealtimeBuffer<bool> emergency_stop_buffer_;

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr command_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr emergency_stop_sub_;
  std::shared_ptr<realtime_tools::RealtimePublisher<nav_msgs::msg::Odometry>> odom_pub_;

  rclcpp::Time last_odom_publish_time_;
};

}  // namespace rbkairos_real_frequency_response
