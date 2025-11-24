#include "planning/planning_node.hpp"

namespace planning
{
namespace
{
// LanePoint(차량 기준 좌표) → geometry_msgs::Point (ROS 좌표)
// x: forward, y: left(+)/right(-)
geometry_msgs::msg::Point to_point(const PlanningNode::LanePoint & lane_pt, double z)
{
  geometry_msgs::msg::Point pt;
  pt.x = lane_pt.y;  // forward
  pt.y = lane_pt.x;  // lateral
  pt.z = z;
  return pt;
}
}  // namespace

// =======================================
// PlanningNode main flow
// 1) /lane/left, /lane/right 차선 메시지를 수신하고 차량 기준 좌표로 변환.
// 2) 좌/우 차선으로부터 중앙선 추정 + 리샘플링 → waypoint/path 생성.
// 3) nav_msgs/Path, MarkerArray 퍼블리시 → RViz에서 차선/경로 확인.
// =======================================

PlanningNode::PlanningNode()
: rclcpp::Node("planning_node")
{
  // ---- 파라미터 로딩: IPM 스케일, 차선 폭, 경로 길이 등 ----
  frame_id_       = declare_parameter("frame_id", "base_link");
  pixel_scale_x_  = declare_parameter("pixel_scale_x", 0.01);    // m per pixel (좌우)
  pixel_scale_y_  = declare_parameter("pixel_scale_y", 0.01);    // m per pixel (전방)
  ipm_height_     = declare_parameter("ipm_height", 640.0);
  ipm_center_x_   = declare_parameter("ipm_center_x", 400.0);
  flip_y_axis_    = declare_parameter("flip_y_axis", true);

  lane_half_width_  = declare_parameter("lane_half_width", 1.75);
  resample_step_    = declare_parameter("resample_step", 0.5);
  max_path_length_  = declare_parameter("max_path_length", 30.0);
  start_offset_y_   = declare_parameter("start_offset_y", 0.0);
  marker_z_         = declare_parameter("marker_z", 0.0);

  lane_timeout_sec_ = declare_parameter("lane_timeout_sec", 0.2);  // 200 ms 기본

  last_left_stamp_  = this->now();
  last_right_stamp_ = this->now();

  auto qos = rclcpp::QoS(rclcpp::KeepLast(10));

  // ---- 차선 토픽 구독 ----
  lane_left_sub_ = create_subscription<perception::msg::Lane>(
    "/lane/left", qos,
    std::bind(&PlanningNode::on_left_lane, this, std::placeholders::_1));

  lane_right_sub_ = create_subscription<perception::msg::Lane>(
    "/lane/right", qos,
    std::bind(&PlanningNode::on_right_lane, this, std::placeholders::_1));

  // ---- Path / Marker 퍼블리셔 ----
  path_pub_ = create_publisher<nav_msgs::msg::Path>("/planning/path", qos);
  marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/planning/markers", qos);

  RCLCPP_INFO(get_logger(), "Planning node ready (frame: %s)", frame_id_.c_str());
}

// 좌측 차선 콜백
void PlanningNode::on_left_lane(const perception::msg::Lane::ConstSharedPtr msg)
{
  latest_left_ = msg;
  // perception::msg::Lane header 안 채워져 있을 가능성 높아서 그냥 수신 시각 사용
  last_left_stamp_ = this->now();
  process_lanes();
}

// 우측 차선 콜백
void PlanningNode::on_right_lane(const perception::msg::Lane::ConstSharedPtr msg)
{
  latest_right_ = msg;
  last_right_stamp_ = this->now();
  process_lanes();
}

// 좌/우 차선 최신값을 사용해 path + markers 생성
void PlanningNode::process_lanes()
{
  const auto now = this->now();
  const rclcpp::Duration timeout = rclcpp::Duration::from_seconds(lane_timeout_sec_);

  std::vector<LanePoint> left_pts;
  std::vector<LanePoint> right_pts;

  // timeout 이내에 들어온 메시지만 유효하게 사용
  if (latest_left_ && (now - last_left_stamp_) <= timeout) {
    left_pts = convert_lane(latest_left_);
  }

  if (latest_right_ && (now - last_right_stamp_) <= timeout) {
    right_pts = convert_lane(latest_right_);
  }

  // 둘 다 유효하지 않으면 path/marker 지우고 리턴
  if (left_pts.empty() && right_pts.empty())
  {
    // Path 비우기
    nav_msgs::msg::Path empty_path;
    empty_path.header.stamp = now;
    empty_path.header.frame_id = frame_id_;
    path_pub_->publish(empty_path);

    // Marker 전부 삭제
    visualization_msgs::msg::MarkerArray del_array;
    del_array.markers.push_back(make_delete_marker(0, "lane_left"));
    del_array.markers.push_back(make_delete_marker(1, "lane_right"));
    del_array.markers.push_back(make_delete_marker(2, "centerline"));
    marker_pub_->publish(del_array);

    return;
  }

  std::vector<LanePoint> centerline;
  if (!build_centerline(left_pts, right_pts, centerline))
  {
    RCLCPP_DEBUG(get_logger(), "Insufficient lane data for path");

    // centerline 이 아예 안 만들어졌으면 centerline marker만 삭제
    visualization_msgs::msg::MarkerArray del_array;
    del_array.markers.push_back(make_delete_marker(2, "centerline"));
    marker_pub_->publish(del_array);
    return;
  }

  publish_path(centerline);
  publish_markers(left_pts, right_pts, centerline);
}

// perception::msg::Lane(IPM 픽셀) → LanePoint(차량 기준 [m])
std::vector<PlanningNode::LanePoint> PlanningNode::convert_lane(
  const perception::msg::Lane::ConstSharedPtr & lane_msg) const
{
  std::vector<LanePoint> out;
  if (!lane_msg) {
    return out;
  }

  out.reserve(lane_msg->lane_points.size());

  for (const auto & pt : lane_msg->lane_points)
  {
    // IPM 이미지 기준 y 픽셀 → 위가 가까운 쪽이 되도록 flip (필요 시)
    double y_pix = flip_y_axis_
      ? (ipm_height_ - static_cast<double>(pt.y))
      : static_cast<double>(pt.y);

    LanePoint lane_pt;

    // 좌우 방향: 왼쪽(+), 오른쪽(-)
    // IPM: pt.x < ipm_center_x_ → 왼쪽
    lane_pt.x = (ipm_center_x_ - static_cast<double>(pt.x)) * pixel_scale_x_;

    // 전방 방향: y 픽셀 → m
    lane_pt.y = y_pix * pixel_scale_y_ + start_offset_y_;

    out.push_back(lane_pt);
  }

  // 전방 방향(y) 순으로 정렬 (보간용)
  std::sort(out.begin(), out.end(),
    [](const LanePoint & a, const LanePoint & b) {
      return a.y < b.y;
    });

  return out;
}

// 특정 전방 거리 longitudinal(y)에 대해 차선 x를 보간
std::optional<double> PlanningNode::sample_lane(
  const std::vector<LanePoint> & lane, double longitudinal) const
{
  if (lane.size() < 2) {
    return std::nullopt;
  }
  if (longitudinal < lane.front().y || longitudinal > lane.back().y) {
    return std::nullopt;
  }

  auto upper = std::lower_bound(
    lane.begin(), lane.end(), longitudinal,
    [](const LanePoint & pt, double value) { return pt.y < value; });

  if (upper == lane.begin()) {
    return upper->x;
  }
  if (upper == lane.end()) {
    return (lane.end() - 1)->x;
  }

  const auto & p1 = *(upper - 1);
  const auto & p2 = *upper;
  const double dy = p2.y - p1.y;
  if (std::abs(dy) < 1e-3) {
    return p1.x;
  }

  const double ratio = (longitudinal - p1.y) / dy;
  return p1.x + ratio * (p2.x - p1.x);
}

// 좌/우 차선으로부터 중앙선 생성
bool PlanningNode::build_centerline(
  const std::vector<LanePoint> & left,
  const std::vector<LanePoint> & right,
  std::vector<LanePoint> & centerline) const
{
  centerline.clear();
  if (left.empty() && right.empty()) {
    return false;
  }

  const double y_end = start_offset_y_ + max_path_length_;

  for (double y = start_offset_y_; y <= y_end; y += resample_step_)
  {
    auto left_x  = sample_lane(left, y);
    auto right_x = sample_lane(right, y);

    if (!left_x && !right_x) {
      continue;
    }

    LanePoint pt;
    pt.y = y;

    if (left_x && right_x) {
      // 양쪽 차선 모두 있을 때 → 평균
      pt.x = (*left_x + *right_x) * 0.5;
    } else if (left_x) {
      // 왼쪽만 있을 때 → lane_half_width_만큼 오른쪽(-)으로 이동
      pt.x = *left_x - lane_half_width_;
    } else {
      // 오른쪽만 있을 때 → lane_half_width_만큼 왼쪽(+)으로 이동
      pt.x = *right_x + lane_half_width_;
    }

    centerline.push_back(pt);
  }

  return !centerline.empty();
}

// 중앙선 → nav_msgs/Path 로 퍼블리시
void PlanningNode::publish_path(const std::vector<LanePoint> & centerline)
{
  nav_msgs::msg::Path path_msg;
  path_msg.header.stamp = now();
  path_msg.header.frame_id = frame_id_;

  for (size_t i = 0; i < centerline.size(); ++i)
  {
    const auto & pt = centerline[i];

    geometry_msgs::msg::PoseStamped pose;
    pose.header = path_msg.header;

    // 좌표계 매핑: x=forward, y=lateral
    pose.pose.position.x = pt.y;
    pose.pose.position.y = pt.x;
    pose.pose.position.z = marker_z_;

    // yaw 계산 (다음 점과의 방향)
    double yaw = 0.0;
    if (i + 1 < centerline.size())
    {
      const auto & next = centerline[i + 1];
      yaw = std::atan2(next.x - pt.x, next.y - pt.y);
    }

    pose.pose.orientation.x = 0.0;
    pose.pose.orientation.y = 0.0;
    pose.pose.orientation.z = std::sin(yaw * 0.5);
    pose.pose.orientation.w = std::cos(yaw * 0.5);

    path_msg.poses.push_back(pose);
  }

  path_pub_->publish(path_msg);
}

// 좌/우/중앙선 → MarkerArray (LINE_STRIP) 로 퍼블리시
void PlanningNode::publish_markers(
  const std::vector<LanePoint> & left,
  const std::vector<LanePoint> & right,
  const std::vector<LanePoint> & centerline)
{
  visualization_msgs::msg::MarkerArray array;

  // 왼쪽 차선
  if (!left.empty()) {
    array.markers.push_back(make_marker(left, 0, "lane_left", 0.2, 0.6, 1.0));
  } else {
    array.markers.push_back(make_delete_marker(0, "lane_left"));
  }

  // 오른쪽 차선
  if (!right.empty()) {
    array.markers.push_back(make_marker(right, 1, "lane_right", 1.0, 0.6, 0.2));
  } else {
    array.markers.push_back(make_delete_marker(1, "lane_right"));
  }

  // 중앙선
  if (!centerline.empty()) {
    array.markers.push_back(make_marker(centerline, 2, "centerline", 0.1, 1.0, 0.2));
  } else {
    array.markers.push_back(make_delete_marker(2, "centerline"));
  }

  marker_pub_->publish(array);
}

// 실제 차선 라인 Marker
visualization_msgs::msg::Marker PlanningNode::make_marker(
  const std::vector<LanePoint> & lane,
  int id,
  const std::string & ns,
  double r, double g, double b) const
{
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = frame_id_;
  marker.header.stamp = now();
  marker.ns = ns;
  marker.id = id;
  marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.scale.x = 0.05;
  marker.color.a = 1.0;
  marker.color.r = r;
  marker.color.g = g;
  marker.color.b = b;
  marker.pose.orientation.w = 1.0;

  marker.points.reserve(lane.size());
  for (const auto & pt : lane) {
    marker.points.push_back(to_point(pt, marker_z_));
  }

  return marker;
}

// 기존 Marker 삭제용
visualization_msgs::msg::Marker PlanningNode::make_delete_marker(
  int id,
  const std::string & ns) const
{
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = frame_id_;
  marker.header.stamp = now();
  marker.ns = ns;
  marker.id = id;
  marker.action = visualization_msgs::msg::Marker::DELETE;
  return marker;
}

}  // namespace planning

// =======================================
// main
// =======================================
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<planning::PlanningNode>());
  rclcpp::shutdown();
  return 0;
}
