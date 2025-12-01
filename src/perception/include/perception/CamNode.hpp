#pragma once

#include "global/global.hpp"
#include <functional>
#include <string>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "rclcpp/qos.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "perception/msg/lane.hpp"

// CameraProcessing 클래스: 이미지 구독, 차선 메시지 발행, 시각화 관리
namespace perception
{
class CameraProcessing : public rclcpp::Node
{
public:
  CameraProcessing();
  ~CameraProcessing() override;

private:
  void on_image(const sensor_msgs::msg::Image::ConstSharedPtr msg);
  void publish_lane_messages();

  std::string window_name_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
  rclcpp::Publisher<perception::msg::Lane>::SharedPtr lane_left_pub_;
  rclcpp::Publisher<perception::msg::Lane>::SharedPtr lane_right_pub_;
};

} // namespace perception
