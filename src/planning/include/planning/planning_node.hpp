#pragma once

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

private:
  // 파라미터
  std::string frame_id_;     // 기본: "base_link" (원하면 launch/yaml에서 "map"으로 override)
  double pixel_scale_x_;     // m per pixel (좌우)
  double pixel_scale_y_;     // m per pixel (전방)
  double ipm_height_;        // IPM 이미지 세로 픽셀 수 (RemapHeight)
  double ipm_center_x_;      // IPM 상에서 차량 중심 x 픽셀
  bool   flip_y_axis_;       // IPM y축 뒤집을지 여부 (위=가까운 쪽으로 맞추기용)

  double lane_half_width_;   // 한 차선 폭의 절반 [m]
  double resample_step_;     // 중앙선 리샘플링 간격 [m]
  double max_path_length_;   // Path 최대 길이 [m]
  double start_offset_y_;    // 차량 기준 y offset [m]
  double marker_z_;          // z 높이 [m] (시각화용)

  // ROS 통신
  rclcpp::Subscription<perception::msg::Lane>::SharedPtr lane_left_sub_;
  rclcpp::Subscription<perception::msg::Lane>::SharedPtr lane_right_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  // 최신 차선 메시지 저장
  perception::msg::Lane::ConstSharedPtr latest_left_;
  perception::msg::Lane::ConstSharedPtr latest_right_;
};

}  // namespace planning
