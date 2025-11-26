#include "control/control_node.hpp"

// =======================================
// ControlNode main flow
// 1) /planning/path 로 전달된 중앙 경로를 Pure Pursuit 로 추종하여 조향각(angular.z)을 계산한다.
// 2) 경로 기울기(= 차선 기울기)를 이용해 "곡률이 클수록(많이 꺾일수록)" 속도를 줄인다.
//    - PID 제어기로 slope(기울기)를 error 로 보고 base_speed 에서 감산.
// 3) 최종 Twist(/cmd_vel)를 발행해 차량 조향 + 속도 제어를 수행한다.
// =======================================

namespace control
{
namespace
{
// ---- 기본 파라미터 (필요하면 launch에서 override 가능) ----

// Pure Pursuit 기본 lookahead 거리 [m] (scale car 기준: 0.2~0.3m 정도)
constexpr double kDefaultLookahead      = 0.25;
constexpr double kDefaultMinLookahead   = 0.15;
constexpr double kDefaultMaxLookahead   = 0.40;

// 속도 명령 범위 [cmd]
//  - 실제 차량에서는  -50 ~ +50 까지 쓸 수 있다고 가정했지만,
//  - 차선 추종에서는 보통 "전진"만 할 거라, 여기서는 0 ~ 50 기준으로 사용.
//  - 필요하면 min_speed 를 음수로 줘서 후진도 가능.
constexpr double kDefaultBaseSpeed      = 40.0;   // 직선에서 목표 속도 (기준)
constexpr double kDefaultMinSpeed       = 10.0;   // 최저 속도
constexpr double kDefaultMaxSpeed       = 50.0;   // 최고 속도 (하드 제한)

// 조향 angular.z 제한 [rad]
//  - 하드웨어 스펙: [-1, +1]
//  - +1 : 우회전, -1 : 좌회전 (ROS 표준과 부호가 반대인 셈)
constexpr double kDefaultMaxAngularZ    = 1.0;

// 곡선에서 속도 줄이기용 PID (입력: slope_abs = |기울기|)
constexpr double kDefaultSpeedKp        = 30.0;   // P 게인
constexpr double kDefaultSpeedKi        = 0.0;    // I 게인 (필요시만)
constexpr double kDefaultSpeedKd        = 2.0;    // D 게인
constexpr double kDefaultIntegralLimit  = 5.0;    // 적분 항 제한

// 속도 제어에 사용할 기울기 측정 구간 [m]
//  - path 전체가 아니라, 차량 앞쪽 일정 거리까지만 보고 곡률을 판단
constexpr double kDefaultSlopeRange     = 0.4;    // 0.4m 구간 안에서 slope 계산
}  // namespace

ControlNode::ControlNode()
: rclcpp::Node("lane_follow_control"),
  // --- Pure Pursuit / Lookahead ---
  lookahead_distance_(declare_parameter("lookahead_distance", kDefaultLookahead)),
  min_lookahead_(declare_parameter("min_lookahead", kDefaultMinLookahead)),
  max_lookahead_(declare_parameter("max_lookahead", kDefaultMaxLookahead)),
  // --- 속도 관련 ---
  base_speed_(declare_parameter("base_speed", kDefaultBaseSpeed)),
  min_speed_(declare_parameter("min_speed", kDefaultMinSpeed)),
  max_speed_(declare_parameter("max_speed", kDefaultMaxSpeed)),
  max_angular_z_(declare_parameter("max_angular_z", kDefaultMaxAngularZ)),
  // --- 곡률 기반 속도 PID ---
  speed_kp_(declare_parameter("slope_speed_kp", kDefaultSpeedKp)),
  speed_ki_(declare_parameter("slope_speed_ki", kDefaultSpeedKi)),
  speed_kd_(declare_parameter("slope_speed_kd", kDefaultSpeedKd)),
  integral_limit_(declare_parameter("slope_integral_limit", kDefaultIntegralLimit)),
  slope_range_(declare_parameter("slope_range", kDefaultSlopeRange)),
  slope_integral_(0.0),
  prev_error_(0.0),
  last_update_time_(this->now())
{
  // ---- 플래닝 경로 구독 ----
  const std::string path_topic = declare_parameter("path_topic", std::string("/planning/path"));
  auto qos = rclcpp::QoS(rclcpp::KeepLast(10));

  path_sub_ = create_subscription<nav_msgs::msg::Path>(
    path_topic, qos,
    std::bind(&ControlNode::on_path, this, std::placeholders::_1));

  // ---- cmd_vel 퍼블리셔 ----
  cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", qos);

  RCLCPP_INFO(get_logger(),
    "Control node ready (lookahead %.2f [m], base speed %.2f [cmd])",
    lookahead_distance_, base_speed_);
}

//==================================================//
// Path 콜백
//  - /planning/path 메시지를 받아서 Pure Pursuit + 속도 제어 수행
//==================================================//
void ControlNode::on_path(const nav_msgs::msg::Path::SharedPtr msg)
{
  if (!msg || msg->poses.empty())
  {
    RCLCPP_DEBUG(get_logger(), "Empty path received, skip control.");
    return;
  }

  // dt 계산 (PID에서 사용)
  const rclcpp::Time now = this->now();
  const double dt = std::max(1e-3, (now - last_update_time_).seconds());
  last_update_time_ = now;

  // ------------------------------------------------------
  // 1) Path 메시지를 내부 2D 포맷(Point2D)으로 변환
  //
  // planning node 에서:
  //   pose.position.x = forward (전방, +앞)
  //   pose.position.y = lateral (좌+ / 우-)
  //
  // 여기서는:
  //   Point2D:
  //     x = lateral
  //     y = forward
  // 로 맞춰서 Pure Pursuit 수식에 사용한다.
  // ------------------------------------------------------
  std::vector<Point2D> path_points;
  path_points.reserve(msg->poses.size());
  for (const auto & pose : msg->poses)
  {
    const double forward = pose.pose.position.x;
    const double lateral = pose.pose.position.y;
    // 차량 기준 앞으로만 사용 (뒤쪽 포인트는 무시)
    if (forward < 0.0) {
      continue;
    }
    path_points.push_back(Point2D{lateral, forward});
  }

  if (path_points.size() < 2)
  {
    RCLCPP_DEBUG(get_logger(), "Too few valid path points (forward>0).");
    return;
  }

  // ------------------------------------------------------
  // 2) Pure Pursuit: lookahead 지점(target) 찾기
  // ------------------------------------------------------
  Point2D target{0.0, 0.0};
  double selected_lookahead = lookahead_distance_;

  if (!compute_lookahead_target(path_points, lookahead_distance_, target, selected_lookahead))
  {
    RCLCPP_DEBUG(get_logger(), "Failed to find lookahead target.");
    return;
  }

  // Pure Pursuit curvature 계산
  //  - 좌표계: 차량기준, x=lateral, y=forward
  //  - 곡률 κ ≈ 2 * x / L^2
  const double curvature = (2.0 * target.x) /
    std::max(1e-3, selected_lookahead * selected_lookahead);

  // ------------------------------------------------------
  // 3) 속도 제어: path 기울기(곡률) 기반으로 속도 줄이기
  // ------------------------------------------------------
  const double slope = estimate_lane_slope(path_points, slope_range_);
  const double speed_cmd = update_speed_command(slope, dt);

  // ------------------------------------------------------
  // 4) 최종 Twist 생성
  //    - linear.x : 속도 명령 (0 ~ 50 사이로 clamping)
  //    - angular.z: 조향 명령 ([-1, 1] 사이)
  //
  // ★ 중요: 차량 스펙
  //   +1 rad  → 우회전
  //   -1 rad  → 좌회전
  //
  //   하지만 Pure Pursuit curvature는 보통:
  //     x>0 (좌측에 목표) → "좌회전" 필요 → κ>0
  //
  //   따라서, "좌회전이 양수"인 세계관과
  //   "우회전이 양수"인 하드웨어 사이 부호가 반대라서
  //   여기서 한 번 -1 을 곱해준다:
  //     angular.z = - κ * v
  // ------------------------------------------------------
  geometry_msgs::msg::Twist cmd;
  cmd.linear.x =
    std::clamp(speed_cmd, 0.0, max_speed_);  // 전진만 사용 (원하면 음수 허용 가능)

  const double raw_ang = -curvature * cmd.linear.x;  // 부호 반전 (좌+ → 우+ 로)
  cmd.angular.z =
    std::clamp(raw_ang, -max_angular_z_, max_angular_z_);

  cmd_pub_->publish(cmd);

  RCLCPP_DEBUG(get_logger(),
    "target=(lat=%.3f, fwd=%.3f) L=%.3f κ=%.3f slope=%.3f speed=%.2f ang=%.2f",
    target.x, target.y, selected_lookahead, curvature, slope, cmd.linear.x, cmd.angular.z);
}

//==================================================//
// Pure Pursuit용 lookahead target 선택
//  - path_points: 차량 기준 (x=lateral, y=forward)
//  - lookahead_distance: 원하는 L
//  - target: 실제로 선택된 점이 들어감
//  - actual_lookahead: 실제 거리 (보통 desired에 가깝게)
//==================================================//
bool ControlNode::compute_lookahead_target(
  const std::vector<Point2D> & path_points,
  double lookahead_distance,
  Point2D & target,
  double & actual_lookahead) const
{
  if (path_points.empty())
  {
    return false;
  }

  const double desired =
    std::clamp(lookahead_distance, min_lookahead_, max_lookahead_);
  const double desired_sq = desired * desired;

  const Point2D * candidate = nullptr;

  for (const auto & pt : path_points)
  {
    // 차량 기준 거리^2 = x^2 + y^2
    const double dist_sq = pt.x * pt.x + pt.y * pt.y;

    // forward(=y)가 너무 작거나 음수인 포인트는 skip 해도 됨
    if (pt.y < 0.0) {
      continue;
    }

    if (dist_sq >= desired_sq)
    {
      candidate = &pt;
      actual_lookahead = std::sqrt(dist_sq);
      break;
    }
  }

  // 원하는 거리만큼 앞에 있는 점을 못 찾으면, 맨 마지막 점을 사용
  if (!candidate)
  {
    const auto & back = path_points.back();
    const double dist = std::hypot(back.x, back.y);
    if (dist < 1e-3)
    {
      return false;
    }
    candidate = &back;
    actual_lookahead = dist;
  }

  target = *candidate;
  return true;
}

//==================================================//
// 경로 기울기(=차선 기울기) 계산
//
// - path_points: 차량 기준 좌표 (x=lateral, y=forward)
// - max_forward_range: 이 값보다 앞쪽까지만 보고 기울기 계산
//
// 구현:
//   1) y ∈ [0, max_forward_range] 구간에 들어가는 포인트만 모아서
//   2) 맨 앞/맨 뒤 포인트를 이용해
//        slope = Δx / Δy
//      계산.
//==================================================//
double ControlNode::estimate_lane_slope(
  const std::vector<Point2D> & path_points,
  double max_forward_range) const
{
  if (path_points.size() < 2)
  {
    return 0.0;
  }

  // 앞쪽 일정 구간만 사용
  std::vector<Point2D> segment;
  segment.reserve(path_points.size());
  for (const auto & pt : path_points)
  {
    if (pt.y < 0.0) {
      continue;
    }
    if (pt.y <= max_forward_range) {
      segment.push_back(pt);
    } else {
      break;  // path 가 y 증가 순으로 들어왔다고 가정
    }
  }

  if (segment.size() < 2)
  {
    // 데이터가 너무 적으면 그냥 전체 path 기준으로 계산
    segment = path_points;
  }

  const auto & first = segment.front();
  const auto & last  = segment.back();

  const double dy = last.y - first.y;
  if (std::abs(dy) < 1e-4)
  {
    return 0.0;
  }

  // slope = Δx / Δy
  return (last.x - first.x) / dy;
}

//==================================================//
// 속도 명령 업데이트
//
// 입력:
//   - slope: 경로 기울기 (좌우 휘어진 정도)
//   - dt   : 경과 시간
//
// 아이디어:
//   - "기울기 절댓값"을 error 로 둔다. (직선일수록 error=0)
//   - PID 로 correction 을 만들고,
//   - base_speed 에서 correction 을 빼서 속도를 줄인다.
//   - 결과를 [min_speed_, max_speed_] 범위로 제한.
//==================================================//
double ControlNode::update_speed_command(double slope, double dt)
{
  const double error = std::abs(slope);  // 직선에서 0, 많이 휘어질수록 커짐

  // 적분 항 업데이트 (anti-windup)
  slope_integral_ += error * dt;
  slope_integral_ =
    std::clamp(slope_integral_, -integral_limit_, integral_limit_);

  const double derivative = (error - prev_error_) / dt;
  prev_error_ = error;

  const double correction =
    speed_kp_ * error +
    speed_ki_ * slope_integral_ +
    speed_kd_ * derivative;

  // base_speed 에서 correction 을 빼서 곡선에서 속도를 줄임
  double cmd = base_speed_ - correction;

  // 하드 제한
  cmd = std::clamp(cmd, min_speed_, max_speed_);
  return cmd;
}

}  // namespace control

//==================================================//
// main
//==================================================//
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<control::ControlNode>());
  rclcpp::shutdown();
  return 0;
}
