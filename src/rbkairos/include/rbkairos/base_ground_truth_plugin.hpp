#ifndef RBKAIROS__BASE_GROUND_TRUTH_PLUGIN_HPP_
#define RBKAIROS__BASE_GROUND_TRUTH_PLUGIN_HPP_

#include <memory>
#include <string>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "mujoco_sim_ros2/mujoco_physics_plugin.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"

namespace rbkairos
{

class BaseGroundTruthPlugin final : public mujoco_sim_ros2::MujocoPhysicsPlugin
{
public:
  BaseGroundTruthPlugin();
  ~BaseGroundTruthPlugin() override = default;

  void Configure(
    rclcpp::Node::SharedPtr &node, rclcpp::NodeOptions cm_node_option, mjModel *model,
    mjData *data) override;
  void Reset(mjModel *model, mjData *data) override;
  void Update(mjModel *model, mjData *data) override;

private:
  template <typename T>
  T declare_or_get_parameter(const std::string &name, const T &default_value)
  {
    if (!node_->has_parameter(name))
    {
      node_->declare_parameter<T>(name, default_value);
    }
    return node_->get_parameter(name).get_value<T>();
  }

  bool resolve_body(mjModel *model);
  bool should_publish(double sim_time) const;
  bool should_print(double sim_time) const;
  rclcpp::Time sim_time_to_ros_time(double sim_time) const;
  double yaw_from_mujoco_quat(const mjtNum quat[4]) const;
  void publish_messages(
    const rclcpp::Time &stamp, const mjtNum pos[3], const mjtNum quat[4],
    const mjtNum world_velocity[6], const mjtNum local_velocity[6]);
  void print_sample(
    const mjtNum pos[3], const mjtNum quat[4], const mjtNum world_velocity[6]) const;

  rclcpp::Node::SharedPtr node_;
  rclcpp::Logger logger_;

  std::string body_name_;
  std::string world_frame_id_;
  std::string child_frame_id_;
  std::string odom_frame_id_;
  std::string pose_topic_;
  std::string twist_topic_;
  std::string odom_topic_;

  bool publish_pose_;
  bool publish_twist_;
  bool publish_odom_;
  bool print_enabled_;

  double publish_rate_hz_;
  double print_rate_hz_;
  double last_publish_time_;
  double last_print_time_;

  mjModel *model_;
  int body_id_;
  bool enabled_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
};

}  // namespace rbkairos

#endif  // RBKAIROS__BASE_GROUND_TRUTH_PLUGIN_HPP_
