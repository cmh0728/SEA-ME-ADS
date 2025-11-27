#pragma once

#include "global/global.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "perception/msg/lane.hpp"

namespace planning
{

class PlanningNode : public rclcpp::Node
{
public:
  struct LanePoint
  {
    double x;  // lateral (left +, right -)  [m]
    double y;  // longitudinal (forward +)   [m]
  };

  PlanningNode();
  void LoadParam();
  
private:
  // 콜백
  void on_left_lane(const perception::msg::Lane::ConstSharedPtr msg);
  void on_right_lane(const perception::msg::Lane::ConstSharedPtr msg);

  // 메인 처리
  void process_lanes();

  // lane msg(IPM 픽셀 좌표) → 차량 로컬 좌표(LanePoint)
  std::vector<LanePoint> convert_lane(
    const perception::msg::Lane::ConstSharedPtr & lane_msg) const;

  // 주어진 y(전방 거리)에서 차선 x(lateral)를 선형보간으로 샘플링
  std::optional<double> sample_lane(
    const std::vector<LanePoint> & lane, double longitudinal) const;

  // 좌/우 차선으로부터 중앙선 생성
  bool build_centerline(
    const std::vector<LanePoint> & left,
    const std::vector<LanePoint> & right,
    std::vector<LanePoint> & centerline) const;

  // nav_msgs/Path 퍼블리시
  void publish_path(const std::vector<LanePoint> & centerline);

  // RViz 시각화용 MarkerArray 퍼블리시
  void publish_markers(
    const std::vector<LanePoint> & left,
    const std::vector<LanePoint> & right,
    const std::vector<LanePoint> & centerline);

  visualization_msgs::msg::Marker make_marker(
    const std::vector<LanePoint> & lane,
    int id,
    const std::string & ns,
    double r, double g, double b) const;

  visualization_msgs::msg::Marker make_delete_marker(
    int id,
    const std::string & ns) const;

private:
  // IPM 월드 영역 정의 [m]
  double x_min_m_;
  double x_max_m_;
  double y_min_m_;
  double y_max_m_;

  // IPM 이미지 크기 [px]
  double ipm_width_;
  double ipm_height_;

  // 기존에 있던 것들
  std::string frame_id_;

  double lane_half_width_;
  double resample_step_;
  double max_path_length_;
  double start_offset_y_;
  double marker_z_;
  double lane_timeout_sec_;
  double origin_offset_x_m_;
  double origin_offset_y_m_;

  // ROS 통신
  rclcpp::Subscription<perception::msg::Lane>::SharedPtr lane_left_sub_;
  rclcpp::Subscription<perception::msg::Lane>::SharedPtr lane_right_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  // 최신 차선 메시지 & 마지막 수신 시각
  perception::msg::Lane::ConstSharedPtr latest_left_;
  perception::msg::Lane::ConstSharedPtr latest_right_;
  rclcpp::Time last_left_stamp_;
  rclcpp::Time last_right_stamp_;
};

}  // namespace planning
