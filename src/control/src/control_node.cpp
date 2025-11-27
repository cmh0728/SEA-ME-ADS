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
  const std::string path_topic = declare_parameter("path_topic", std::string("/planning/path"));
  auto qos = rclcpp::QoS(rclcpp::KeepLast(10));

  path_sub_ = create_subscription<nav_msgs::msg::Path>(
    path_topic, qos,
    std::bind(&ControlNode::on_path, this, std::placeholders::_1));

  cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", qos);

  // ★ 타겟 포인트용 Marker 퍼블리셔
  target_marker_pub_ =
    create_publisher<visualization_msgs::msg::Marker>("/control/lookahead_target", 1);

  RCLCPP_INFO(get_logger(),
    "Control node ready (lookahead %.2f m, base speed %.2f)",
    lookahead_distance_, base_speed_);
}

//================================================== on_path func ==================================================//
void ControlNode::on_path(const nav_msgs::msg::Path::SharedPtr msg)
{
  if (!msg || msg->poses.empty())
  {
    return;
  }

  const rclcpp::Time now = this->now();
  const double dt = std::max(1e-3, (now - last_update_time_).seconds());
  last_update_time_ = now;

  // 1) Path → (x=lateral, y=forward)
  std::vector<Point2D> path_points;
  path_points.reserve(msg->poses.size());
  for (const auto & pose : msg->poses)
  {
    Point2D pt{pose.pose.position.y, pose.pose.position.x};
    path_points.push_back(pt);
  }

  if (path_points.empty())
  {
    return;
  }

  // 2) 경로 기울기 계산 (직선 vs 곡선 판단용)
  const double slope = estimate_lane_slope(path_points);

  // 3) 속도 명령 계산 (이미 있던 함수 그대로 사용)
  const double speed_cmd = update_speed_command(slope, dt);

  // 4) 속도에 따른 동적 lookahead 계산
  //    - 직선(속도↑) → lookahead 크게 (max_lookahead_)
  //    - 곡선(속도↓) → lookahead 작게 (min_lookahead_)
  double speed_norm = 0.0;
  if (max_speed_ > min_speed_)
  {
    speed_norm = (speed_cmd - min_speed_) / (max_speed_ - min_speed_);
  }
  speed_norm = std::clamp(speed_norm, 0.0, 1.0);

  // 동적으로 사용할 lookahead
  const double dynamic_lookahead =
    min_lookahead_ + (max_lookahead_ - min_lookahead_) * speed_norm;

  // 5) dynamic_lookahead를 사용해서 Pure Pursuit 타겟 선택
  Point2D target{0.0, 0.0};
  double selected_lookahead = 0.0;
  if (!compute_lookahead_target(path_points, dynamic_lookahead, target, selected_lookahead))
  {
    return;
  }

  // RViz 타겟 시각화
  publish_target_marker(target, msg->header.frame_id);

  // 6) Pure Pursuit 곡률 계산
  const double curvature =
    (2.0 * target.x) / std::max(1e-3, selected_lookahead * selected_lookahead);

  // 7) 조향 게인 & 부호 보정
  constexpr double kSteerGain = 0.03;  // 이미 쓰던 값

  double steer_cmd = -kSteerGain * curvature * speed_cmd;
  steer_cmd = std::clamp(steer_cmd, -max_angular_z_, max_angular_z_);

  // 8) 최종 Twist 구성
  geometry_msgs::msg::Twist cmd;
  cmd.linear.x  = std::clamp(speed_cmd, -max_speed_, max_speed_);
  cmd.angular.z = steer_cmd;

  cmd_pub_->publish(cmd);

  RCLCPP_DEBUG(
    get_logger(),
    "PP target=(%.3f, %.3f) L(desired)=%.3f L(actual)=%.3f slope=%.3f v=%.2f steer=%.3f",
    target.x, target.y, dynamic_lookahead, selected_lookahead,
    slope, cmd.linear.x, cmd.angular.z);
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


void ControlNode::publish_target_marker(const Point2D & target, const std::string & frame_id)
{
  visualization_msgs::msg::Marker marker;

  marker.header.frame_id = frame_id;     // path와 같은 frame 사용 (보통 base_link)
  marker.header.stamp    = this->now();
  marker.ns   = "lookahead_target";
  marker.id   = 0;
  marker.type = visualization_msgs::msg::Marker::SPHERE;
  marker.action = visualization_msgs::msg::Marker::ADD;

  // Point2D: x = lateral, y = forward
  // Marker 좌표: x = forward, y = lateral 이라서 다시 매핑
  marker.pose.position.x = target.y;   // forward
  marker.pose.position.y = target.x;   // lateral
  marker.pose.position.z = 0.05;       // 바닥에서 조금 띄우기

  marker.pose.orientation.w = 1.0;

  // 점 크기
  marker.scale.x = 0.06;
  marker.scale.y = 0.06;
  marker.scale.z = 0.06;

  // 빨간색
  marker.color.a = 1.0;
  marker.color.r = 1.0;
  marker.color.g = 0.0;
  marker.color.b = 0.0;

  target_marker_pub_->publish(marker);
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
