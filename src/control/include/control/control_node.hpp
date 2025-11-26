#ifndef CONTROL__CONTROL_NODE_HPP_
#define CONTROL__CONTROL_NODE_HPP_

#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/twist.hpp"

namespace control
{

// 간단한 2D 포인트 구조체
// x: lateral (좌/우), y: forward (전방)
struct Point2D
{
  double x;
  double y;
};

class ControlNode : public rclcpp::Node
{
public:
  explicit ControlNode();

private:
  // /planning/path 콜백
  void on_path(const nav_msgs::msg::Path::SharedPtr msg);

  // Pure Pursuit용 lookahead target 선택
  //  - path_points : 차량 기준 좌표 (x=lateral, y=forward)
  //  - lookahead_distance : 원하는 L
  //  - target : 선택된 포인트
  //  - actual_lookahead : 실제 거리
  bool compute_lookahead_target(
    const std::vector<Point2D> & path_points,
    double lookahead_distance,
    Point2D & target,
    double & actual_lookahead) const;

  // 경로 기울기(차선 기울기) 계산
  //  - max_forward_range [m] 까지의 구간만 보고 slope 추정
  double estimate_lane_slope(
    const std::vector<Point2D> & path_points,
    double max_forward_range) const;

  // 곡선에서 속도 줄이기 위한 PID
  //  - slope: 경로 기울기
  //  - dt   : 샘플링 시간
  double update_speed_command(double slope, double dt);

  // ======== ROS 통신 객체 ======== //
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;

  // ======== Pure Pursuit / Lookahead 파라미터 ======== //
  double lookahead_distance_;
  double min_lookahead_;
  double max_lookahead_;

  // ======== 속도 관련 파라미터 (cmd_vel.linear.x) ======== //
  double base_speed_;     // 직선 기준 속도
  double min_speed_;      // 최저 속도
  double max_speed_;      // 최고 속도

  // ======== 조향 제한 (cmd_vel.angular.z) ======== //
  double max_angular_z_;  // [-max, +max], +는 우회전, -는 좌회전

  // ======== 곡선 구간 속도 제어용 PID ======== //
  double speed_kp_;
  double speed_ki_;
  double speed_kd_;
  double integral_limit_; // 적분 항 제한
  double slope_range_;    // 기울기 측정에 사용할 전방 거리 [m]

  double slope_integral_; // ∫ |slope| dt
  double prev_error_;     // 이전 |slope|
  rclcpp::Time last_update_time_;
};

}  // namespace control

#endif  // CONTROL__CONTROL_NODE_HPP_
