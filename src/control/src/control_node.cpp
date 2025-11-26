#include "control/control_node.hpp"

#include <algorithm>
#include <cmath>
#include <functional>

// =======================================
// ControlNode main flow
// 1) /planning/path 로 전달된 중앙 경로를 Pure Pursuit 로 추종하여 조향각(ang.z)을 계산.
// 2) 경로 기울기를 이용해 속도 PID 보정을 수행하고, 곡선에서는 감속.
// 3) 최종 Twist(/cmd_vel)를 발행해 차량 조향 + 속도 제어.
// =======================================

namespace control
{
namespace
{
// ----- 스케일 카 기준 튜닝 값 -----
// IPM 상에서 forward 범위가 0.4~0.7m 정도라서 lookahead도 그 근처로 잡음
constexpr double kDefaultLookahead     = 0.40;  // 기본 lookahead (m)
constexpr double kDefaultMinLookahead  = 0.30;  // 최소 lookahead
constexpr double kDefaultMaxLookahead  = 0.60;  // 최대 lookahead

// speed: cmd_vel.linear.x = 0 ~ 50 근처 사용
constexpr double kDefaultBaseSpeed     = 25.0;  // 직선 기준 속도
constexpr double kDefaultMinSpeed      =  5.0;  // 너무 느리면 제어 불안정하니 5 이상
constexpr double kDefaultMaxSpeed      = 50.0;  // 하드웨어 상한

// steer: cmd_vel.angular.z = -1 ~ +1  ( -1 좌, +1 우 )
constexpr double kDefaultMaxAngular    = 1.0;

// 곡선 구간 속도 제어용 PID 파라미터 (필요하면 나중에 튜닝)
constexpr double kDefaultSpeedKp       = 4.0;
constexpr double kDefaultSpeedKi       = 0.0;
constexpr double kDefaultSpeedKd       = 0.2;
constexpr double kDefaultIntegralLimit = 5.0;

// 조향 민감도 (curvature → steer 로 보낼 때 gain)
//   - 0.1 ~ 0.5 사이에서 시작해보고 튜닝하면 됨
constexpr double kSteerGain            = 0.4;
}  // namespace

//================================================== ctor ==================================================//

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
  // ---- 플래닝 경로 구독 ----
  const std::string path_topic = declare_parameter("path_topic", std::string("/planning/path"));
  auto qos = rclcpp::QoS(rclcpp::KeepLast(10));

  path_sub_ = create_subscription<nav_msgs::msg::Path>(
    path_topic, qos,
    std::bind(&ControlNode::on_path, this, std::placeholders::_1));

  cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", qos);

  RCLCPP_INFO(get_logger(),
    "Control node ready (lookahead %.2f m, base speed %.2f)",
    lookahead_distance_, base_speed_);
}

//================================================== on_path ==================================================//

void ControlNode::on_path(const nav_msgs::msg::Path::SharedPtr msg)
{
  if (!msg || msg->poses.empty())
  {
    return;
  }

  const rclcpp::Time now = this->now();
  const double dt = std::max(1e-3, (now - last_update_time_).seconds());
  last_update_time_ = now;

  // Path 메시지를 Pure Pursuit 계산에 쓰기 위해 (x=횡, y=종) 포맷으로 변환
  //  - planning node 에서:
  //      pose.position.x = forward (전방, +)
  //      pose.position.y = lateral (좌우, 왼+ / 오-)
  //  → 여기서는 Point2D{x=lateral, y=forward} 로 다시 저장
  std::vector<Point2D> path_points;
  path_points.reserve(msg->poses.size());
  for (const auto & pose : msg->poses)
  {
    Point2D pt{pose.pose.position.y, pose.pose.position.x};
    path_points.push_back(pt);
  }

  // Pure Pursuit target 계산
  Point2D target{0.0, 0.0};
  double selected_lookahead = lookahead_distance_;
  if (!compute_lookahead_target(path_points, lookahead_distance_, target, selected_lookahead))
  {
    return;
  }

  // --- 조향 curvature 계산 ---
  //   curvature = 2 * x / L^2
  //   x: lateral offset (좌 +, 우 -), L: lookahead
  const double curvature =
    (2.0 * target.x) / std::max(1e-3, selected_lookahead * selected_lookahead);

  // --- 경로 기울기(곡률 느낌) 계산: 곡선에서 속도 줄이기 위함 ---
  const double slope = estimate_lane_slope(path_points);
  const double speed_cmd = update_speed_command(slope, dt);

  // ---- 최종 명령 생성 ----
  geometry_msgs::msg::Twist cmd = build_cmd(curvature, speed_cmd);

  // ★ 조향만 별도로 계산 ★
  //
  // - 차량 좌표계:
  //     x>0 : 왼쪽, x<0 : 오른쪽
  // - curvature>0 : 왼쪽으로 휘는 곡선
  // - 하지만 하드웨어:
  //     cmd.angular.z = +1  → 우회전
  //     cmd.angular.z = -1  → 좌회전
  //
  // → 부호를 한 번 뒤집어서 사용해야 함.
  double raw_steer = -kSteerGain * curvature;  // 부호 뒤집기 + gain 적용
  raw_steer = std::clamp(raw_steer, -max_angular_z_, max_angular_z_);
  cmd.angular.z = raw_steer;

  cmd_pub_->publish(cmd);

  RCLCPP_DEBUG(get_logger(),
    "target=(%.2f, %.2f) L=%.2f curvature=%.3f slope=%.3f speed=%.2f steer=%.2f",
    target.x, target.y, selected_lookahead, curvature, slope, cmd.linear.x, cmd.angular.z);
}

//================================================== compute_lookahead_target ==================================================//

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

//================================================== estimate_lane_slope ==================================================//

double ControlNode::estimate_lane_slope(const std::vector<Point2D> & path_points) const
{
  if (path_points.size() < 2)
  {
    return 0.0;
  }

  // 간단히: 맨 앞 포인트와 맨 뒤 포인트로 전체 기울기 계산
  //   slope = Δx / Δy (전방 기준)
  const auto & first = path_points.front();
  const auto & last  = path_points.back();
  const double dy = last.y - first.y;
  if (std::abs(dy) < 1e-3)
  {
    return 0.0;
  }
  return (last.x - first.x) / dy;
}

//================================================== update_speed_command ==================================================//

double ControlNode::update_speed_command(double slope, double dt)
{
  // slope가 클수록 (더 많이 기울어질수록) → 곡선 구간 → 속도 줄이기
  const double error = std::abs(slope);  // 직선(기울기 0)을 target으로 보는 개념

  slope_integral_ = std::clamp(
    slope_integral_ + error * dt, -integral_limit_, integral_limit_);

  const double derivative = (error - prev_slope_) / dt;
  prev_slope_ = error;

  // PID 보정값이 클수록 속도를 더 깎음
  const double correction =
    speed_kp_ * error + speed_ki_ * slope_integral_ + speed_kd_ * derivative;

  // base_speed_ 에서 correction 만큼 빼고, [min_speed_, max_speed_]로 제한
  const double command =
    std::clamp(base_speed_ - correction, min_speed_, max_speed_);

  return command;
}

//================================================== build_cmd ==================================================//

geometry_msgs::msg::Twist ControlNode::build_cmd(double /*curvature*/, double speed) const
{
  geometry_msgs::msg::Twist cmd;

  // speed: 0 ~ 50 근처로 clamping
  cmd.linear.x = std::clamp(speed, min_speed_, max_speed_);

  // 조향은 on_path()에서 따로 설정하므로 여기서는 0으로 초기화
  cmd.angular.z = 0.0;
  return cmd;
}

}  // namespace control

//================================================== main ==================================================//

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<control::ControlNode>());
  rclcpp::shutdown();
  return 0;
}
