#ifndef CONTROL__CONTROL_NODE_HPP_
#define CONTROL__CONTROL_NODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/path.hpp"
#include <vector>

namespace control
{
// 계획된 경로를 받아 Pure Pursuit로 조향하고 차선 기울기 기반 PID로 속도를 조절합니다.
class ControlNode : public rclcpp::Node
{
public:
  ControlNode();

private:
  struct Point2D
  {
    double x;
    double y;
  };

  void on_path(const nav_msgs::msg::Path::SharedPtr msg);
  bool compute_lookahead_target(const std::vector<Point2D> & path_points,
                                double lookahead_distance,
                                Point2D & target,
                                double & actual_lookahead) const;
  double estimate_lane_slope(const std::vector<Point2D> & path_points) const;
  double update_speed_command(double slope, double dt);
  geometry_msgs::msg::Twist build_cmd(double curvature, double speed) const;

  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;

  double lookahead_distance_;
  double min_lookahead_;
  double max_lookahead_;
  double base_speed_;
  double min_speed_;
  double max_speed_;
  double max_angular_z_;
  double speed_kp_;
  double speed_ki_;
  double speed_kd_;
  double integral_limit_;

  double slope_integral_;
  double prev_slope_;
  rclcpp::Time last_update_time_;
};
}  // namespace control

#endif  // CONTROL__CONTROL_NODE_HPP_
