#include "perception/perception_node.hpp"

#include <chrono>
#include <cmath>

using namespace std::chrono_literals;

namespace perception
{
// 100ms마다 가상의 장애물 거리와 차선 오프셋을 출판해 이후 단계에서 사용할 수 있게 합니다.
PerceptionNode::PerceptionNode()
: rclcpp::Node("perception_node")
{
  publisher_ = create_publisher<sea_interfaces::msg::PerceptionData>(
    "/perception/data", rclcpp::QoS(10));

  timer_ = create_wall_timer(100ms, [this]() { publish_mock_measurement(); });

  RCLCPP_INFO(get_logger(), "Perception node initialized");
}

// 간단한 사인 곡선을 사용해 시간에 따라 변화하는 측정값을 생성합니다.
void PerceptionNode::publish_mock_measurement()
{
  static float phase = 0.0F;
  phase += 0.1F;

  auto message = sea_interfaces::msg::PerceptionData();
  message.stamp = now();
  message.sensor_frame = "lidar_front";
  message.obstacle_distance = 15.0F + 5.0F * std::sin(phase);
  message.lane_offset = 0.2F * std::sin(phase * 0.5F);
  message.ego_velocity = 12.0F;

  publisher_->publish(message);
  RCLCPP_DEBUG(get_logger(), "Published perception data: obstacle %.2f m", message.obstacle_distance);
}
}  // namespace perception

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<perception::PerceptionNode>());
  rclcpp::shutdown();
  return 0;
}
