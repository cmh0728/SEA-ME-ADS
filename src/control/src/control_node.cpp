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

// 나중에 yaml 파일로 정리 
// ----- pure pursuit params -----
constexpr double kDefaultLookahead     = 0.45;  // 기본 lookahead (m)
constexpr double kDefaultMinLookahead  = 0.40;  // 최소 lookahead
constexpr double kDefaultMaxLookahead  = 0.70;  // 최대 lookahead
constexpr double kDefualtCarL          = 0.26;  // 차량 축간 거리 (m)

// speed: cmd_vel.linear.x = 0 ~ 50 근처 사용
constexpr double kDefaultBaseSpeed     = 20.0;  // 직선 기준 속도
constexpr double kDefaultMinSpeed      = 10.0;  // 너무 느리면 제어 불안정하니 10 이상
constexpr double kDefaultMaxSpeed      = 50.0;  // 하드웨어 상한

// steer
constexpr double kDefaultMaxAngular    = 1.0;

// 속도 제어용 PID 파라미터 
constexpr double kDefaultSpeedKp       = 10.0;
constexpr double kDefaultSpeedKi       = 0.001;
constexpr double kDefaultSpeedKd       = 0.7;
constexpr double kDefaultIntegralLimit = 5.0; // 적분 항 클램프 한계

// 조향 민감도 (curvature → steer 로 보낼 때 gain)
constexpr double kSteerGain            = 2.293;
// 조향 신뢰도 체크 
constexpr double kDefaultMaxSteerRate  = 8.0; // rad/s, 한 프레임당 변화율 제한
constexpr double kDefaultMaxSteerJump  = 1.5; // 변화량 제한값



}  // namespace

//================================================== ctor ==================================================//

ControlNode::ControlNode(): rclcpp::Node("control_node"),
  // Ld
  lookahead_distance_(declare_parameter("lookahead_distance", kDefaultLookahead)),
  min_lookahead_(declare_parameter("min_lookahead", kDefaultMinLookahead)),
  max_lookahead_(declare_parameter("max_lookahead", kDefaultMaxLookahead)),
  car_L(declare_parameter("car_L", kDefualtCarL)),

  // speed / steer limits
  base_speed_(declare_parameter("base_speed", kDefaultBaseSpeed)),
  min_speed_(declare_parameter("min_speed", kDefaultMinSpeed)),
  max_speed_(declare_parameter("max_speed", kDefaultMaxSpeed)),
  max_angular_z_(declare_parameter("max_angular_z", kDefaultMaxAngular)),
  // PID params
  speed_kp_(declare_parameter("slope_speed_kp", kDefaultSpeedKp)),
  speed_ki_(declare_parameter("slope_speed_ki", kDefaultSpeedKi)),
  speed_kd_(declare_parameter("slope_speed_kd", kDefaultSpeedKd)),
  integral_limit_(declare_parameter("slope_integral_limit", kDefaultIntegralLimit)),
  slope_integral_(0.0),
  prev_slope_(0.0),
  last_update_time_(this->now()),
    // ===== 조향 필터용 파라미터 & 상태 =====
  max_steer_rate_(declare_parameter("max_steer_rate", kDefaultMaxSteerRate)),
  max_steer_jump_(declare_parameter("max_steer_jump", kDefaultMaxSteerJump)),
  prev_steer_cmd_(0.0),
  has_prev_steer_(false)
{
  const std::string path_topic = declare_parameter("path_topic", std::string("/planning/path"));
  auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();

  // path sub
  path_sub_ = create_subscription<nav_msgs::msg::Path>(
    path_topic, qos,
    std::bind(&ControlNode::on_path, this, std::placeholders::_1));

  // cmd pub
  cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", qos);

  // ld pub
  target_marker_pub_ =
    create_publisher<visualization_msgs::msg::Marker>("/control/lookahead_target", 1);

  // log 
  RCLCPP_INFO(get_logger(),
    "Control node ready (lookahead %.2f m, base speed %.2f)",
    lookahead_distance_, base_speed_);
}

//================================================== on_path func ==================================================//
// 경로 수신 콜백
void ControlNode::on_path(const nav_msgs::msg::Path::SharedPtr msg)
{
  if (!msg || msg->poses.empty())
  {
    return;
  }

  const rclcpp::Time now = this->now();
  const double dt = std::max(1e-3, (now - last_update_time_).seconds()); // planning/path토픽의 hz의 역수 (30hz ~45)
  last_update_time_ = now;

  // Path → (x=lateral, y=forward)
  std::vector<Point2D> path_points; // 차량 기준 좌표계 포인트 저장 vector  
  path_points.reserve(msg->poses.size()); // 메모리 미리 할당
  for (const auto & pose : msg->poses)
  {
    Point2D pt{pose.pose.position.y, pose.pose.position.x}; // 좌표 변환, Point2D 에 저장
    path_points.push_back(pt); // path_points 에 추가
  }

  if (path_points.empty())
  {
    return;
  }

  // 경로 기울기 계산 , 곡선구간 0.4 언저리 
  const double slope = estimate_lane_slope(path_points);
  // std::cout << "estimated slope: " << slope << std::endl;

  // 속도 명령 계산 
  const double speed_cmd = update_speed_command(slope, dt);

  // ====== Ld: 경로 내에서 차량 중심(0,0)으로부터 가장 먼 점 사용 ======
  Point2D target{0.0, 0.0};
  double selected_lookahead = 0.0;   // = |target| (거리)

  // lookahead_distance 인자는 더 이상 사용하지 않으므로 0.0 전달
  if (!compute_lookahead_target(path_points, 0.0, target, selected_lookahead))
  {
    return;
  }


  double error = target.x; // 오른쪽으로 치우치면 양수값, 왼쪽으로 치우치면 음수값 (scale : m)
  // std::cout << "error : " << error << std::endl;

  // RViz 타겟 시각화
  publish_target_marker(target, msg->header.frame_id);

  // Pure Pursuit 곡률 계산
  const double curvature = (2.0 * target.x) / std::max(1e-3, selected_lookahead * selected_lookahead);
  
  // std::cout << "curvature : " << curvature << std::endl;
  // 조향각 계산 --> slope에 따라서 다른 게인 적용 
  double raw_steer = std::atan(kDefualtCarL * curvature); 

  // 곡선구간 steer 보정 (민감도 up)
  const double abs_slope = std::abs(slope);
  // std::cout << "abs_slope : " << abs_slope << std::endl; 
  double gain_factor = kSteerGain;

  // 곡선 정도에 따라 추가 게인
  // slope 대략 값: 직선 ~0.01, 완만 곡선 ~0.1~0.3, 강한 곡선 ~0.4 이상이라고 했으니까 그 기준으로.
  if (abs_slope > 0.30) {
    // 많이 꺾인 곡선 → 게인 크게
    gain_factor *= 2;   // 필요하면 1.8, 2.0 등으로 더 키워도 됨
  } else if (abs_slope > 0.1) {
    // 중간 정도 곡선
    gain_factor *= 1.2;
  } else {
    // 거의 직선
    gain_factor *= 1.0;
  }
  
  // steer 보정 
  double steer_cmd = raw_steer * gain_factor;

  // 1차 안전 클램프
  steer_cmd = std::clamp(steer_cmd, -max_angular_z_, max_angular_z_);

  // === 이전 프레임과 비교해서 이상치/과도한 변화 방지 ===
  steer_cmd = filter_steering(steer_cmd, dt);

  // 최종 안전 클램프 (필터 후에도 범위 보장)
  steer_cmd = std::clamp(steer_cmd, -max_angular_z_, max_angular_z_);


  // 최종 Twist 구성
  geometry_msgs::msg::Twist cmd;
  cmd.linear.x  = std::clamp(speed_cmd, -max_speed_, max_speed_);
  cmd.angular.z = steer_cmd;

  cmd_pub_->publish(cmd);

  RCLCPP_DEBUG(
    get_logger(),
    "PP target=(%.3f, %.3f) L=%.3f slope=%.3f v=%.2f steer=%.3f",
    target.x, target.y, selected_lookahead,
    slope, cmd.linear.x, cmd.angular.z);

}


//================================================== compute_lookahead_target func ==================================================//
// Pure Pursuit 타겟 선택
bool ControlNode::compute_lookahead_target(const std::vector<Point2D> & path_points,
                                           double /*lookahead_distance*/,
                                           Point2D & target,
                                           double & actual_lookahead) const
{
  if (path_points.empty())
  {
    return false;
  }

  const Point2D* candidate = nullptr;
  double max_dist_sq = 0.0;

  // 차량 앞쪽만 고려(원하면 pt.y > 0.0 조건은 빼도 됨)
  constexpr double kMinForward = 0.0;  // 완전 앞쪽만 쓰고 싶으면 0.05 같은 값으로 올리기

  for (const auto & pt : path_points)
  {
    // 차량 뒤쪽/정확히 중심은 스킵
    if (pt.y <= kMinForward)
    {
      continue;
    }

    const double dist_sq = pt.x * pt.x + pt.y * pt.y;

    if (!candidate || dist_sq > max_dist_sq)
    {
      candidate    = &pt;
      max_dist_sq  = dist_sq;
    }
  }

  // 전방에 후보가 하나도 없을 때 fallback: 마지막 점 사용
  if (!candidate)
  {
    const auto & back = path_points.back();
    const double dist_sq = back.x * back.x + back.y * back.y;

    if (dist_sq < 1e-6)
    {
      return false;
    }

    target = back;
    actual_lookahead = std::sqrt(dist_sq);
    return true;
  }

  target = *candidate;
  actual_lookahead = std::sqrt(max_dist_sq);
  return true;
}


//================================================== estimate_lane_slope func ==================================================//
// 경로 기울기 간단 추정 --> lane coefficient  계산한거 넘겨주는 코드 추가 
double ControlNode::estimate_lane_slope(const std::vector<Point2D> & path_points) const
{
  if (path_points.size() < 2)
  {
    return 0.0;
  }

  // 맨 앞 포인트와 맨 뒤 포인트로 전체 기울기 계산
  //   slope = Δx / Δy (전방 기준)
  const auto & first = path_points.front();
  const auto & last  = path_points.back();
  const double dy = last.y - first.y;
  if (std::abs(dy) < 1e-3)
  {
    return 0.0;
  }
  double slope = (last.x - first.x) / dy;
  // std::cout << "slope: " << slope << std::endl; // 곡선구간 0.4정도, 직선구간 0.01~근처 
  return slope;
}

//================================================== update_speed_command func ==================================================//
// 경로 기울기에 따른 속도 PID 보정
double ControlNode::update_speed_command(double slope, double dt)
{
  // slope가 클수록 (더 많이 기울어질수록) → 곡선 구간 → 속도 줄이기
  const double error = std::abs(slope);  // 직선에 가까울수록 0에 가까워짐 

  // 적분 항 업데이트 (클램프)
  slope_integral_ = std::clamp(slope_integral_ + error * dt, -integral_limit_, integral_limit_);

  // 미분 항 계산
  const double derivative = (error - prev_slope_) / dt;
  prev_slope_ = error;

  // PID 계산 --> 직선이면 correction 값이 거의 0 
  const double correction = speed_kp_ * error + speed_ki_ * slope_integral_ + speed_kd_ * derivative;

  // base_speed_ 에서 correction 만큼 빼고, [min_speed_, max_speed_]로 제한
  const double command = std::clamp(base_speed_ - correction, min_speed_, base_speed_);

  return command;
}

//================================================== build_cmd func ==================================================//
// ros topic /cmd_vel 용 Twist 메시지 생성
geometry_msgs::msg::Twist ControlNode::build_cmd(double /*curvature*/, double speed) const
{
  geometry_msgs::msg::Twist cmd;

  // 최저,최대 속도 설정
  cmd.linear.x = std::clamp(speed, min_speed_, max_speed_);

  // 조향은 on_path()에서 따로 설정하므로 여기서는 0으로 초기화
  cmd.angular.z = 0.0;
  return cmd;
}

//================================================== publish_target_marker func ==================================================//
// ld marker pub
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

//================================================== filter_steering func ==================================================//
double ControlNode::filter_steering(double raw_steer, double dt)
{
  // 최초 프레임이면 그냥 통과
  if (!has_prev_steer_) {
    prev_steer_cmd_ = raw_steer;
    has_prev_steer_ = true;
    return raw_steer;
  }

  const double delta = raw_steer - prev_steer_cmd_; // 조향값 변화량 

  // 완전 말도 안되는 조향값 감지 → 이번 프레임은 버리고 이전 값 유지
  if (std::abs(delta) > max_steer_jump_)
  {
    RCLCPP_WARN(
      get_logger(),
      "Reject steering outlier: raw=%.3f, prev=%.3f (|Δ|=%.3f > %.3f)",
      raw_steer, prev_steer_cmd_, delta, max_steer_jump_);
    return prev_steer_cmd_;
  }

  // 정상 범위 내에서는 변화율 제한 (rate limiting) , dt는 0.033~0.025
  const double max_delta = max_steer_rate_ * dt; // 이번 프레임에서 허용 가능한 최대 변화량 --> 0.12정도 
  double limited_delta = delta;
  if (std::abs(limited_delta) > max_delta)
  {
    limited_delta = (limited_delta > 0.0 ? 1.0 : -1.0) * max_delta;
  }

  const double filtered = prev_steer_cmd_ + limited_delta;

  prev_steer_cmd_ = filtered;
  return filtered;
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
