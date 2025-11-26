#pragma once

#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/path.hpp"

namespace control
{

// 2D 포인트 (차량 기준 좌표계)
// x: lateral  (왼쪽 +, 오른쪽 -)
// y: forward  (앞 +)
struct Point2D
{
  double x;
  double y;
};

class ControlNode : public rclcpp::Node
{
public:
  ControlNode();

private:
  // ---- 콜백 ----
  void on_path(const nav_msgs::msg::Path::SharedPtr msg);

  // ---- Pure Pursuit 관련 ----
  // path_points: 차량 기준 좌표계 (x=lateral, y=forward)
  // lookahead_distance: 원하는 L
  // target: 선택된 lookahead 타겟
  // actual_lookahead: 실제 타겟까지 거리
  bool compute_lookahead_target(
    const std::vector<Point2D> & path_points,
    double lookahead_distance,
    Point2D & target,
    double & actual_lookahead) const;

  // 경로 전체 기울기(곡선 정도)를 간단히 추정
  //  - path_points.front() ~ back() 를 사용해 slope = Δx / Δy
  double estimate_lane_slope(
    const std::vector<Point2D> & path_points) const;

  // 차선 기울기(곡률 느낌)에 따라 속도를 PID로 보정
  //  - slope: estimate_lane_slope() 결과
  //  - dt   : 이전 업데이트 이후 경과 시간 [s]
  double update_speed_command(double slope, double dt);

  // Twist 메시지 생성 (현재 구현에서는 speed만 세팅, steer는 on_path에서 계산)
  geometry_msgs::msg::Twist build_cmd(double curvature, double speed) const;

  // ---- 파라미터 / 상태 ----
  // Pure Pursuit 관련
  double lookahead_distance_;
  double min_lookahead_;
  double max_lookahead_;

  // 속도/조향 한계
  double base_speed_;
  double min_speed_;
  double max_speed_;
  double max_angular_z_;

  // 속도 PID (곡선에서 감속)
  double speed_kp_;
  double speed_ki_;
  double speed_kd_;
  double integral_limit_;

  double slope_integral_;
  double prev_slope_;        // 이전 slope (derivative 계산용)
  rclcpp::Time last_update_time_;

  // ---- ROS 통신 ----
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
};

}  // namespace control
