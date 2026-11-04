#ifndef RBKAIROS__TRACTION_TENDON_OBSERVER_PLUGIN_HPP_
#define RBKAIROS__TRACTION_TENDON_OBSERVER_PLUGIN_HPP_

#include <array>
#include <cstddef>
#include <memory>
#include <string>

#include "geometry_msgs/msg/wrench_stamped.hpp"
#include "mujoco_sim_ros2/mujoco_physics_plugin.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace rbkairos
{

class TractionTendonObserverPlugin final : public mujoco_sim_ros2::MujocoPhysicsPlugin
{
public:
  static constexpr int kStateSchemaVersion = 1;
  static constexpr std::size_t kStateFieldCount = 17;

  TractionTendonObserverPlugin();
  ~TractionTendonObserverPlugin() override = default;

  void Configure(
    rclcpp::Node::SharedPtr & node, rclcpp::NodeOptions cm_node_option, mjModel * model,
    mjData * data) override;
  void Reset(mjModel * model, mjData * data) override;
  void PostUpdate(mjModel * model, mjData * data) override;

private:
  struct TendonSample
  {
    double sim_time_s{0.0};
    double tendon_length_m{0.0};
    double tendon_length0_m{0.0};
    double lower_limit_m{0.0};
    double upper_limit_m{0.0};
    double extension_over_upper_limit_m{0.0};
    double tension_n{0.0};
    bool limit_active{false};
    std::array<double, 3> force_n{{0.0, 0.0, 0.0}};
    std::array<double, 3> attach_pos_m{{0.0, 0.0, 0.0}};
    std::array<double, 3> anchor_pos_m{{0.0, 0.0, 0.0}};
  };

  template<typename T>
  T declare_or_get_parameter(const std::string & name, const T & default_value)
  {
    if (!node_->has_parameter(name)) {
      node_->declare_parameter<T>(name, default_value);
    }
    return node_->get_parameter(name).get_value<T>();
  }

  bool resolve_model_objects(const mjModel * model);
  bool compute_sample(
    const mjModel * model, const mjData * data, TendonSample & sample) const;
  void reset_publish_schedule();
  bool should_publish(double sim_time_s) const;
  bool should_print(double sim_time_s) const;
  rclcpp::Time sim_time_to_ros_time(double sim_time_s) const;
  void publish_sample(const TendonSample & sample);
  void print_sample(const TendonSample & sample) const;

  rclcpp::Node::SharedPtr node_;
  rclcpp::Logger logger_;

  bool enabled_;
  bool print_enabled_;
  double publish_rate_hz_;
  double print_rate_hz_;
  double last_publish_time_s_;
  double last_print_time_s_;

  std::string tendon_name_;
  std::string sensor_name_;
  std::string attach_site_name_;
  std::string anchor_site_name_;
  std::string state_topic_;
  std::string wrench_topic_;
  std::string frame_id_;

  const mjModel * model_;
  int tendon_id_;
  int sensor_id_;
  int attach_site_id_;
  int anchor_site_id_;

  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr state_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_publisher_;
};

}  // namespace rbkairos

#endif  // RBKAIROS__TRACTION_TENDON_OBSERVER_PLUGIN_HPP_
