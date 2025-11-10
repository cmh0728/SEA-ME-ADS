#include "control/control_node.hpp"

#include <algorithm>

namespace control
{
namespace
{
constexpr double kDefaultKp = 100;     // 픽셀 에러를 각속도로 변환하는 기본 비례 이득
constexpr double kDefaultKi = 0.01;
constexpr double kDefaultKd = 0.5;
constexpr double kDefaultLinearSpeed = 15.0;  // 차량 프로토콜 기준 +15가 기본 주행속도 --> 차후에 곡률에 따라 속도 조절 기능 추가 
constexpr double kDefaultMaxAngular = 1.0; //조향 최댓값
constexpr double kDefaultMaxIntegral = 1.0;
constexpr double kDefaultPixelToMeter = 0.35 / 542 ;  // user tunable scale
constexpr double kDefaultWatchdogSec = 0.5;
}  // namespace

ControlNode::ControlNode()
: rclcpp::Node("lane_follow_control"),
  integral_error_(0.0),
  prev_error_(0.0),
  last_stamp_(this->now()),
  watchdog_timeout_(rclcpp::Duration::from_seconds(kDefaultWatchdogSec))
{
  // PID 및 차량 주행 관련 기본 파라미터 선언 
  kp_ = declare_parameter("kp", kDefaultKp);
  ki_ = declare_parameter("ki", kDefaultKi);
  kd_ = declare_parameter("kd", kDefaultKd);
  linear_speed_ = declare_parameter("linear_speed", kDefaultLinearSpeed);
  max_angular_z_ = declare_parameter("max_angular_z", kDefaultMaxAngular);
  max_integral_ = declare_parameter("max_integral", kDefaultMaxIntegral);
  pixel_to_meter_ = declare_parameter("pixel_to_meter", kDefaultPixelToMeter);
  const double watchdog = declare_parameter("watchdog_timeout", kDefaultWatchdogSec);
  watchdog_timeout_ = rclcpp::Duration::from_seconds(watchdog);

  // /cmd_vel 퍼블리셔와 차선 오프셋 구독 설정
  cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", rclcpp::QoS(10));
  offset_sub_ = create_subscription<std_msgs::msg::Float32>(
    "/lane/center_offset", rclcpp::QoS(10),
    std::bind(&ControlNode::on_offset, this, std::placeholders::_1));

  RCLCPP_INFO(get_logger(), "PID controller initialized (kp=%.4f ki=%.4f kd=%.4f)", kp_, ki_, kd_);
}

void ControlNode::reset_if_timeout(const rclcpp::Time & now)
{
  // 일정 시간 이상 갱신이 없으면 적분/미분 항을 초기화해 급격한 제어를 방지
  if ((now - last_stamp_) > watchdog_timeout_) {
    integral_error_ = 0.0;
   prev_error_ = 0.0;
  }
}

void ControlNode::on_offset(const std_msgs::msg::Float32::SharedPtr msg)
{
  const rclcpp::Time now = this->now();
  reset_if_timeout(now);

  // dt 계산 (0으로 나누기 방지를 위해 최소값 보장)
  const double dt = std::max(1e-3, (now - last_stamp_).seconds());
  last_stamp_ = now;

  // 오프셋이 양수면 차량이 차선 중앙보다 오른쪽에 있음 (픽셀 → 미터 변환)--> - 조향 필요 
  const double error_px = static_cast<double>(msg->data);
  const double error_m = error_px * pixel_to_meter_;

  // PID 적분/미분 항 계산 및 클램프
  integral_error_ = std::clamp(integral_error_ + error_m * dt, -max_integral_, max_integral_);
  const double derivative = (error_m - prev_error_) / dt;
  prev_error_ = error_m;

  // PID 합산 후 각속도 제한
  double angular_z = kp_ * error_m + ki_ * integral_error_ + kd_ * derivative;
  angular_z = std::clamp(angular_z, -max_angular_z_, max_angular_z_);

  // 최종 Twist 메시지 구성 후 퍼블리시
  geometry_msgs::msg::Twist cmd;
  cmd.linear.x = std::clamp(linear_speed_, -50.0, 50.0);  // 차량 규격 범위 [-50, 50]
  cmd.angular.z = angular_z;
  cmd_pub_->publish(cmd);

  RCLCPP_DEBUG(get_logger(),
    "PID cmd: err_px=%.2f err_m=%.3f ang=%.3f integ=%.3f deriv=%.3f",
    error_px, error_m, angular_z, integral_error_, derivative);
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
