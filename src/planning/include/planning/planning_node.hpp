#ifndef PLANNING__PLANNING_NODE_HPP_
#define PLANNING__PLANNING_NODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "nav_msgs/msg/path.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "perception/msg/lane.hpp"
#include <optional>
#include <string>
#include <vector>

namespace planning
{

class PlanningNode : public rclcpp::Node
{
public:
  struct LanePoint
  {
    double x{};  // lateral (m)
    double y{};  // longitudinal (m)
  };

  PlanningNode();

private:

  void on_left_lane(const perception::msg::Lane::ConstSharedPtr msg);
  void on_right_lane(const perception::msg::Lane::ConstSharedPtr msg);
  void process_lanes();
  std::vector<LanePoint> convert_lane(const perception::msg::Lane::ConstSharedPtr & lane_msg) const;
  std::optional<double> sample_lane(const std::vector<LanePoint> & lane, double longitudinal) const;
  bool build_centerline(const std::vector<LanePoint> & left,
                        const std::vector<LanePoint> & right,
                        std::vector<LanePoint> & centerline) const;
  void publish_path(const std::vector<LanePoint> & centerline);
  void publish_markers(const std::vector<LanePoint> & left,
                       const std::vector<LanePoint> & right,
                       const std::vector<LanePoint> & centerline);
  visualization_msgs::msg::Marker make_marker(const std::vector<LanePoint> & lane,
                                              int id,
                                              const std::string & ns,
                                              double r, double g, double b) const;

  rclcpp::Subscription<perception::msg::Lane>::SharedPtr lane_left_sub_;
  rclcpp::Subscription<perception::msg::Lane>::SharedPtr lane_right_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  perception::msg::Lane::ConstSharedPtr latest_left_;
  perception::msg::Lane::ConstSharedPtr latest_right_;

  std::string frame_id_;
  double pixel_scale_x_;
  double pixel_scale_y_;
  double ipm_height_;
  double ipm_center_x_;
  bool flip_y_axis_;
  double lane_half_width_;
  double resample_step_;
  double max_path_length_;
  double start_offset_y_;
  double marker_z_;
};

}  // namespace planning

#endif  // PLANNING__PLANNING_NODE_HPP_
