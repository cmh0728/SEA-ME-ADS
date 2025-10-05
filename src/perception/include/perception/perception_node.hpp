#ifndef PERCEPTION__PERCEPTION_NODE_HPP_
#define PERCEPTION__PERCEPTION_NODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "sea_interfaces/msg/perception_data.hpp"

namespace perception
{
// 모의 센서 데이터를 발행해 파이프라인의 입력을 제공하는 노드입니다.
class PerceptionNode : public rclcpp::Node
{
public:
  PerceptionNode();

private:
  void publish_mock_measurement();

  rclcpp::Publisher<sea_interfaces::msg::PerceptionData>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};
}  // namespace perception

#endif  // PERCEPTION__PERCEPTION_NODE_HPP_
