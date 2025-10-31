#ifndef PERCEPTION__PERCEPTION_NODE_HPP_
#define PERCEPTION__PERCEPTION_NODE_HPP_

#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"

namespace perception
{
// RealSense 이미지 토픽을 구독해 OpenCV 창으로 출력하는 간단한 지각 노드입니다.
class PerceptionNode : public rclcpp::Node
{
public:
  PerceptionNode();
  ~PerceptionNode() override;

private:
  void on_image(const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg);

  std::string window_name_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_subscription_;
};
}  // namespace perception

#endif  // PERCEPTION__PERCEPTION_NODE_HPP_
