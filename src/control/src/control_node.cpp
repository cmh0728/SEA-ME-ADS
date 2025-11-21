#include "control/control_node.hpp"

#include <algorithm>
#include <cmath>
#include <functional>

namespace control
{
namespace
{
constexpr double kDefaultLookahead = 3.0;
constexpr double kDefaultMinLookahead = 1.5;
constexpr double kDefaultMaxLookahead = 6.0;
constexpr double kDefaultBaseSpeed = 10.0;
constexpr double kDefaultMinSpeed = 3.0;
constexpr double kDefaultMaxSpeed = 20.0;
constexpr double kDefaultMaxAngular = 1.2;
constexpr double kDefaultSpeedKp = 4.0;
constexpr double kDefaultSpeedKi = 0.0;
constexpr double kDefaultSpeedKd = 0.2;
constexpr double kDefaultIntegralLimit = 5.0;
}  // namespace

ControlNode::ControlNode()
: rclcpp::Node("lane_follow_control"),
  lookahead_distance_(declare_parameter("lookahead_distance", kDefaultLookahead)),
  min_lookahead_(declare_parameter("min_lookahead", kDefaultMinLookahead)),
  max_lookahead_(declare_parameter("max_lookahead", kDefaultMaxLookahead)),
  base_speed_(declare_parameter("base_speed", kDefaultBaseSpeed)),
  min_speed_(declare_parameter("min_speed", kDefaultMinSpeed)),
  max_speed_(declare_parameter("max_speed", kDefaultMaxSpeed)),
  max_angular_z_(declare_parameter("max_angular_z", kDefaultMaxAngular)),
  speed_kp_(declare_parameter("slope_speed_kp", kDefaultSpeedKp)),
  speed_ki_(declare_parameter("slope_speed_ki", kDefaultSpeedKi)),
  speed_kd_(declare_parameter("slope_speed_kd", kDefaultSpeedKd)),
  integral_limit_(declare_parameter("slope_integral_limit", kDefaultIntegralLimit)),
  slope_integral_(0.0),
  prev_slope_(0.0),
  last_update_time_(this->now())
{
  const std::string path_topic = declare_parameter("path_topic", std::string("/planning/path"));
  auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
  path_sub_ = create_subscription<nav_msgs::msg::Path>(
    path_topic, qos,
    std::bind(&ControlNode::on_path, this, std::placeholders::_1));

  cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", qos);

  RCLCPP_INFO(get_logger(),
    "Control node ready (lookahead %.2f m, base speed %.2f m/s)",
    lookahead_distance_, base_speed_);
}

void ControlNode::on_path(const nav_msgs::msg::Path::SharedPtr msg)
{
  if (!msg || msg->poses.empty())
  {
    return;
  }

  const rclcpp::Time now = this->now();
  const double dt = std::max(1e-3, (now - last_update_time_).seconds());
  last_update_time_ = now;

  std::vector<Point2D> path_points;
  path_points.reserve(msg->poses.size());
  for (const auto & pose : msg->poses)
  {
    Point2D pt{pose.pose.position.y, pose.pose.position.x};
    path_points.push_back(pt);
  }

  Point2D target{0.0, 0.0};
  double selected_lookahead = lookahead_distance_;
  if (!compute_lookahead_target(path_points, lookahead_distance_, target, selected_lookahead))
  {
    return;
  }

  const double curvature = (2.0 * target.x) / std::max(1e-3, selected_lookahead * selected_lookahead);
  const double slope = estimate_lane_slope(path_points);
  const double speed_cmd = update_speed_command(slope, dt);
  const double angular_velocity = std::clamp(curvature * speed_cmd, -max_angular_z_, max_angular_z_);

  geometry_msgs::msg::Twist cmd = build_cmd(curvature, speed_cmd);
  cmd.angular.z = angular_velocity;
  cmd_pub_->publish(cmd);

  RCLCPP_DEBUG(get_logger(),
    "Pure pursuit target=(%.2f, %.2f) lookahead=%.2f slope=%.3f speed=%.2f ang=%.2f",
    target.x, target.y, selected_lookahead, slope, speed_cmd, cmd.angular.z);
}

bool ControlNode::compute_lookahead_target(const std::vector<Point2D> & path_points,
                                           double lookahead_distance,
                                           Point2D & target,
                                           double & actual_lookahead) const
{
  if (path_points.empty())
  {
    return false;
  }

  const double min_l = min_lookahead_;
  const double max_l = max_lookahead_;
  const double desired = std::clamp(lookahead_distance, min_l, max_l);

  const double desired_sq = desired * desired;
  const Point2D * candidate = nullptr;
  for (const auto & pt : path_points)
  {
    const double dist_sq = pt.x * pt.x + pt.y * pt.y;
    if (dist_sq >= desired_sq)
    {
      candidate = &pt;
      actual_lookahead = std::sqrt(dist_sq);
      break;
    }
  }

  if (!candidate)
  {
    candidate = &path_points.back();
    actual_lookahead = std::hypot(candidate->x, candidate->y);
    if (actual_lookahead < 1e-3)
    {
      return false;
    }
  }

  target = *candidate;
  return true;
}

double ControlNode::estimate_lane_slope(const std::vector<Point2D> & path_points) const
{
  if (path_points.size() < 2)
  {
    return 0.0;
  }
  const auto & first = path_points.front();
  const auto & last = path_points.back();
  const double dy = last.y - first.y;
  if (std::abs(dy) < 1e-3)
  {
    return 0.0;
  }
  return (last.x - first.x) / dy;
}

double ControlNode::update_speed_command(double slope, double dt)
{
  slope_integral_ = std::clamp(slope_integral_ + slope * dt, -integral_limit_, integral_limit_);
  const double derivative = (slope - prev_slope_) / dt;
  prev_slope_ = slope;

  const double correction = speed_kp_ * slope + speed_ki_ * slope_integral_ + speed_kd_ * derivative;
  const double command = std::clamp(base_speed_ - correction, min_speed_, max_speed_);
  return command;
}

geometry_msgs::msg::Twist ControlNode::build_cmd(double curvature, double speed) const
{
  geometry_msgs::msg::Twist cmd;
  cmd.linear.x = std::clamp(speed, min_speed_, max_speed_);
  cmd.angular.z = std::clamp(curvature * cmd.linear.x, -max_angular_z_, max_angular_z_);
  return cmd;
}

}  // namespace control

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<control::ControlNode>());
  rclcpp::shutdown();
  return 0;
}


// speed 는 +- 50 까지, (linear x ) , steer는 +- 1 (angular z)(라디안값) +1 이 우회전, -1이 좌회전 
