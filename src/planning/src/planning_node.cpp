#include "planning/planning_node.hpp"

namespace planning
{
namespace
{
// LanePoint(차량 기준 좌표계) → geometry_msgs::Point (Marker에서 사용할 좌표)
// LanePoint:
//   x: lateral  (왼쪽 + / 오른쪽 -)
//   y: forward  (앞쪽 +)
// ROS Point:
//   x: forward, y: left/right 로 쓰겠다는 약속
geometry_msgs::msg::Point to_point(const PlanningNode::LanePoint & lane_pt, double z)
{
  geometry_msgs::msg::Point pt;
  pt.x = lane_pt.y;  // forward  (차량 전방)
  pt.y = lane_pt.x;  // lateral  (차량 좌/우)
  pt.z = z;          // 시각화용 z (보통 0 또는 약간 띄우기)
  return pt;
}
}  // namespace

// =======================================
// PlanningNode main flow
// 1) /lane/left, /lane/right 차선 메시지(IPM 픽셀 좌표)를 수신
// 2) IPM 픽셀 → 차량 기준 [m] 좌표로 변환 (convert_lane)
// 3) 좌/우 차선으로부터 중앙선(centerline) 생성 (build_centerline)
// 4) nav_msgs/Path + MarkerArray 퍼블리시 → RViz에서 확인
// =======================================

PlanningNode::PlanningNode()
: rclcpp::Node("planning_node")
{
  // ------------------------------------------------------
  // [1] IPM 변환 범위 파라미터
  //
  //  - IPM을 만들 때 Python에서 썼던 X_MIN/X_MAX, Y_MIN/Y_MAX, W/H_ipm 값과
  //    "반드시" 동일해야 한다.
  //
  //  - 좌표계 정의 (현재는 카메라 중심 기준으로 정의한다고 가정):
  //      X (전방, forward): 카메라 기준 앞(+), 뒤(-)   [m]
  //      Y (좌우, lateral): 카메라 기준 왼(+), 오른(-) [m]
  //
  //  - 주어진 조건:
  //      · IPM 세로 방향(이미지 상단/하단)이
  //        카메라 중심에서
  //          하단 = 0.42 m (가까운 쪽)
  //          상단 = 0.73 m (먼  쪽)
  //
  //    즉,
  //      r = 0            → X = 0.73 m (이미지 상단, 먼 쪽)
  //      r = ipm_height-1 → X = 0.42 m (이미지 하단, 가까운 쪽)
  //
  //  - 이를 위해:
  //      x_min_m_ = 0.42 (near)
  //      x_max_m_ = 0.73 (far)
  //    으로 두고, 아래 convert_lane()에서
  //      X = x_max_m_ - (x_max_m_ - x_min_m_) * (r / (H-1))
  //    형태로 매핑할 것이다.
  // ------------------------------------------------------
  x_min_m_    = declare_parameter("x_min_m", 0.42);     // near (이미지 하단)  0.42 m
  x_max_m_    = declare_parameter("x_max_m", 0.73);     // far  (이미지 상단)  0.73 m
  y_min_m_    = declare_parameter("y_min_m", -0.25);    // 예: 오른쪽 -0.25 m
  y_max_m_    = declare_parameter("y_max_m",  0.25);    // 예: 왼쪽  +0.25 m
  ipm_width_  = declare_parameter("ipm_width",  400.0); // IPM 가로 픽셀 수 (W)
  ipm_height_ = declare_parameter("ipm_height", 320.0); // IPM 세로 픽셀 수 (H)

  // ------------------------------------------------------
  // [1-1] 원점 오프셋 (IPM 좌표계 원점 → 차량 중심까지 거리)
  //
  //  - 지금 x_min_m_/x_max_m_/y_min_m_/y_max_m_는 "카메라 중심" 기준이라고 가정.
  //  - 나중에 차량 중심(base_link) 기준으로 쓰고 싶으면,
  //      origin_offset_x_m_ / origin_offset_y_m_
  //    를 이용해서 평행 이동할 수 있다.
  //
  //  - 정의:
  //      origin_offset_x_m_ : IPM 좌표계의 (0,0)에서
  //                           "앞쪽(+)/뒤쪽(-)"으로 차량 중심까지 거리 [m]
  //      origin_offset_y_m_ : IPM 좌표계의 (0,0)에서
  //                           "왼쪽(+)/오른쪽(-)"으로 차량 중심까지 거리 [m]
  //
  //  - LanePoint로 쓸 때는:
  //      X_vehicle = X_raw - origin_offset_x_m_
  //      Y_vehicle = Y_raw - origin_offset_y_m_
  //    처럼 원점을 차량 중심으로 평행 이동해 준다.
  //  - 지금은 카메라 기준 그대로 쓰기 위해 0.0으로 두었다.
  // ------------------------------------------------------
  origin_offset_x_m_ = declare_parameter("origin_offset_x_m", 0.0);  // forward offset
  origin_offset_y_m_ = declare_parameter("origin_offset_y_m", 0.0);  // lateral offset

  // ------------------------------------------------------
  // [2] Planning / Visualization 관련 파라미터
  // ------------------------------------------------------

  // frame_id_:
  //   - convert_lane에서 만드는 좌표는 "어떤 기준"에서의 2D 평면 좌표.
  //   - origin_offset_*_m_를 적절히 넣으면 "차량 중심(base_link)" 기준이 된다.
  //   - base_link를 TF로 이미 구성했다면 "base_link"가 자연스럽다.
  frame_id_       = declare_parameter("frame_id", "base_link");

  // lane_half_width_:
  //   - 중앙선 계산에서, 한쪽 차선만 보일 때 사용하는 차선 반폭 [m]
  //   - ex) scale car의 실제 차선폭이 0.35 m 정도면 0.175로 설정
  lane_half_width_  = declare_parameter("lane_half_width", 0.175);

  // resample_step_:
  //   - centerline을 몇 m 간격으로 샘플링할지
  //   - 세로 범위가 0.42 ~ 0.73 (약 0.31 m)이므로,
  //     0.02 m면 대략 15개 포인트 정도.
  resample_step_    = declare_parameter("resample_step", 0.02);  // 2 cm 간격

  // max_path_length_:
  //   - start_offset_y_에서 시작해서 몇 m까지 centerline을 만들지
  //   - 카메라 기준 0.42 ~ 0.73 m 구간을 보고 있다면,
  //     start_offset_y_를 0.42, max_path_length_를 0.31 정도로 두면
  //     대략 0.42 ~ 0.73 m까지만 사용하게 된다.
  //   - 여기서는 기본값을 0.31으로 맞춰둠 (필요하면 parameter로 조정).
  max_path_length_  = declare_parameter("max_path_length", 0.31);

  // start_offset_y_:
  //   - path 시작 전방 거리 [m]
  //   - 카메라 기준 0.42 m부터 보이는 영역이라면 0.42로 설정하는 게 자연스럽다.
  start_offset_y_   = declare_parameter("start_offset_y", 0.42);

  // marker_z_:
  //   - RViz에서 시각화할 때, z 높이 (0.0이면 바닥에 붙음)
  marker_z_         = declare_parameter("marker_z", 0.0);

  // lane_timeout_sec_:
  //   - 이 시간 이상 오래된 lane 메시지는 무시
  //   - Lane이 끊겼을 때 path/marker를 지우기 위해 사용
  lane_timeout_sec_ = declare_parameter("lane_timeout_sec", 0.2);

  last_left_stamp_  = this->now();
  last_right_stamp_ = this->now();

  auto qos = rclcpp::QoS(rclcpp::KeepLast(10));

  // ------------------------------------------------------
  // [3] 차선 토픽 구독
  //   - perception 노드에서 publish하는 /lane/left, /lane/right를 구독
  //   - perception::msg::Lane 안에 lane_points: IPM 이미지 상의 픽셀 좌표 리스트
  // ------------------------------------------------------
  lane_left_sub_ = create_subscription<perception::msg::Lane>(
    "/lane/left", qos,
    std::bind(&PlanningNode::on_left_lane, this, std::placeholders::_1));

  lane_right_sub_ = create_subscription<perception::msg::Lane>(
    "/lane/right", qos,
    std::bind(&PlanningNode::on_right_lane, this, std::placeholders::_1));

  // ------------------------------------------------------
  // [4] Path / Marker 퍼블리셔
  // ------------------------------------------------------
  path_pub_ = create_publisher<nav_msgs::msg::Path>("/planning/path", qos);
  marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/planning/markers", qos);

  RCLCPP_INFO(get_logger(), "Planning node ready (frame: %s)", frame_id_.c_str());
}

//################################################## on_left_lane func ##################################################//

// 좌측 차선 콜백
// - latest_left_ 에 마지막 메시지 저장
// - 타임스탬프 갱신 후 process_lanes() 호출
void PlanningNode::on_left_lane(const perception::msg::Lane::ConstSharedPtr msg)
{
  latest_left_ = msg;
  // perception::msg::Lane header 안 채워져 있을 가능성 높아서 그냥 수신 시각 사용
  last_left_stamp_ = this->now();
  process_lanes();
}

//################################################## on_right_lane func ##################################################//

// 우측 차선 콜백
void PlanningNode::on_right_lane(const perception::msg::Lane::ConstSharedPtr msg)
{
  latest_right_ = msg;
  last_right_stamp_ = this->now();
  process_lanes();
}

//################################################## process_lanes func ##################################################//

// 좌/우 차선 최신값을 사용해 centerline + markers 생성
void PlanningNode::process_lanes()
{
  const auto now = this->now();
  const rclcpp::Duration timeout = rclcpp::Duration::from_seconds(lane_timeout_sec_);

  std::vector<LanePoint> left_pts;
  std::vector<LanePoint> right_pts;

  // timeout 이내에 들어온 메시지만 유효하게 사용 (너무 오래된 차선 정보는 버림)
  if (latest_left_ && (now - last_left_stamp_) <= timeout) {
    left_pts = convert_lane(latest_left_);
  }

  if (latest_right_ && (now - last_right_stamp_) <= timeout) {
    right_pts = convert_lane(latest_right_);
  }

  // 좌/우 둘 다 유효하지 않으면 Path/Marker 모두 삭제 후 리턴
  if (left_pts.empty() && right_pts.empty())
  {
    // Path 비우기
    nav_msgs::msg::Path empty_path;
    empty_path.header.stamp = now;
    empty_path.header.frame_id = frame_id_;
    path_pub_->publish(empty_path);

    // Marker 전부 삭제 (id 0,1,2)
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

  // centerline 으로 path + markers 퍼블리시
  publish_path(centerline);
  publish_markers(left_pts, right_pts, centerline);
}

//################################################## convert_lane func ##################################################//

// perception::msg::Lane(IPM 픽셀) → LanePoint(차량/카메라 기준 [m])
//
// Pixel 기준:
//   c = pt.x (col) : 0 ~ ipm_width_-1   (왼 → 오)
//   r = pt.y (row) : 0 ~ ipm_height_-1  (위 → 아래)
//
// World 기준:
//   X: 전방 (앞 +, 뒤 -)
//   Y: 좌우 (왼 +, 오른 -)
//
// 주어진 조건을 반영:
//   - r = 0           → X = 0.73 m (이미지 상단, 먼 쪽)
//   - r = ipm_height-1→ X = 0.42 m (이미지 하단, 가까운 쪽)
//
// 이를 만족시키기 위해:
//
//   X_raw = x_max_m_ - (x_max_m_ - x_min_m_) * (r / (ipm_height_ - 1))
//
//   (x_min_m_ = 0.42, x_max_m_ = 0.73 일 때)
//
//   r = 0        → X_raw = x_max_m_ = 0.73
//   r = H - 1    → X_raw = x_min_m_ = 0.42
//
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
    // IPM 픽셀 좌표 (col=c, row=r)
    const double c = static_cast<double>(pt.x);  // 0 ~ ipm_width_-1
    const double r = static_cast<double>(pt.y);  // 0 ~ ipm_height_-1

    // --- 전방 방향 X_raw (m) ---
    const double X_raw =
      x_max_m_ - (x_max_m_ - x_min_m_) * (r / (ipm_height_ - 1.0));

    // --- 좌우 방향 Y_raw (m) ---
    // c = 0          -> Y_max (왼쪽)
    // c = W_ipm - 1  -> Y_min (오른쪽)
    const double Y_raw =
      y_max_m_ - (y_max_m_ - y_min_m_) * (c / (ipm_width_ - 1.0));

    // --- 원점 오프셋 보정 (카메라 → 차량 중심 등으로 평행 이동) ---
    const double X = X_raw - origin_offset_x_m_;  // forward
    const double Y = Y_raw - origin_offset_y_m_;  // lateral

    LanePoint lane_pt;
    lane_pt.x = Y;  // lateral (왼쪽 +, 오른쪽 -)
    lane_pt.y = X;  // forward (앞 +)

    out.push_back(lane_pt);
  }

  // 전방 방향(y=forward) 기준 오름차순 정렬
  //   → 이후 보간(sample_lane)에서 사용
  std::sort(out.begin(), out.end(),
    [](const LanePoint & a, const LanePoint & b) {
      return a.y < b.y;
    });

  return out;
}

//################################################## sample_lane func ##################################################//

// 특정 전방 거리 longitudinal(y)에 대해 차선의 x(lateral)를 선형 보간하여 얻기
//   - lane: convert_lane()에서 생성된 LanePoint 리스트 (y 오름차순 정렬)
//   - longitudinal: 조회하고 싶은 전방 거리 [m]
std::optional<double> PlanningNode::sample_lane(
  const std::vector<LanePoint> & lane, double longitudinal) const
{
  if (lane.size() < 2) {
    return std::nullopt;
  }
  if (longitudinal < lane.front().y || longitudinal > lane.back().y) {
    // 요청한 y가 lane 데이터 범위 밖이면 보간 불가
    return std::nullopt;
  }

  // lower_bound로 longitudinal 이상인 첫 점을 찾음
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
    // y가 거의 같으면 그냥 p1.x 사용
    return p1.x;
  }

  // [p1.y, p2.y] 구간에서 선형 보간
  const double ratio = (longitudinal - p1.y) / dy;
  return p1.x + ratio * (p2.x - p1.x);
}

//################################################## build_centerline func ##################################################//

// 좌/우 차선으로부터 중앙선(centerline) 생성
//
// - left, right는 각각 LanePoint 리스트 (기준 좌표계 [m])
// - y(start_offset_y_ ~ y_end) 구간을 일정 간격(resample_step_)으로 스캔하면서
//   해당 y에서의 left_x, right_x를 sample_lane()으로 얻고,
//   둘 다 있으면 평균, 한쪽만 있으면 lane_half_width_로 보정.
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
      // 이 y 위치에서는 두 차선 모두 데이터 없음 → centerline 포인트 생략
      continue;
    }

    LanePoint pt;
    pt.y = y;  // forward

    if (left_x && right_x) {
      // 양쪽 차선 모두 있을 때 → x 좌표 평균
      pt.x = (*left_x + *right_x) * 0.5;
    } else if (left_x) {
      // 왼쪽 차선만 있을 때 → lane_half_width_만큼 오른쪽(-)으로 이동
      pt.x = *left_x - lane_half_width_;
    } else {
      // 오른쪽 차선만 있을 때 → lane_half_width_만큼 왼쪽(+)으로 이동
      pt.x = *right_x + lane_half_width_;
    }

    centerline.push_back(pt);
  }

  return !centerline.empty();
}

//################################################## vis : publish_path func ##################################################//

// 중앙선(centerline) → nav_msgs::Path 로 퍼블리시
//
// - Path의 각 포인트는 PoseStamped
// - position.x: forward, position.y: lateral 로 매핑
// - orientation: 다음 포인트와의 방향(yaw)을 이용해 계산
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
    pose.pose.position.x = pt.y;  // forward
    pose.pose.position.y = pt.x;  // lateral
    pose.pose.position.z = marker_z_;

    // yaw 계산 (다음 점과의 방향)
    double yaw = 0.0;
    if (i + 1 < centerline.size())
    {
      const auto & next = centerline[i + 1];
      // (Δx = next.y - pt.y, Δy = next.x - pt.x)
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

//################################################## vis : publish_markers func ##################################################//

// 좌/우 차선 + 중앙선 → MarkerArray (LINE_STRIP) 로 퍼블리시
//
// - left/right/centerline 각각 다른 색/ID로 표시
void PlanningNode::publish_markers(
  const std::vector<LanePoint> & left,
  const std::vector<LanePoint> & right,
  const std::vector<LanePoint> & centerline)
{
  visualization_msgs::msg::MarkerArray array;

  // 왼쪽 차선 marker
  if (!left.empty()) {
    array.markers.push_back(make_marker(left, 0, "lane_left", 0.2, 0.6, 1.0));
  } else {
    array.markers.push_back(make_delete_marker(0, "lane_left"));
  }

  // 오른쪽 차선 marker
  if (!right.empty()) {
    array.markers.push_back(make_marker(right, 1, "lane_right", 1.0, 0.6, 0.2));
  } else {
    array.markers.push_back(make_delete_marker(1, "lane_right"));
  }

  // 중앙선 marker
  if (!centerline.empty()) {
    array.markers.push_back(make_marker(centerline, 2, "centerline", 0.1, 1.0, 0.2));
  } else {
    array.markers.push_back(make_delete_marker(2, "centerline"));
  }

  marker_pub_->publish(array);
}

//################################################## vis : make_marker func ##################################################//

// 실제 차선 / 중앙선 LineStrip Marker 생성
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
  marker.scale.x = 0.05;  // 선 두께
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

//################################################## vis : make_delete_marker func ##################################################//

// 기존 Marker 삭제용 (id/namespace로 해당 marker 제거)
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

//################################################## main func ##################################################//

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<planning::PlanningNode>());
  rclcpp::shutdown();
  return 0;
}
