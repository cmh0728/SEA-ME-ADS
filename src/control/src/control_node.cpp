#include "control/control_node.hpp"

#include <algorithm>
#include <functional>

namespace control
{
// 제어 노드는 계획된 속도와 차선 목표를 받아 스로틀·브레이크·조향 명령을 생성합니다.
ControlNode::ControlNode()
: rclcpp::Node("control_node")
{
  command_pub_ = create_publisher<sea_interfaces::msg::ControlCommand>(
    "/control/command", rclcpp::QoS(10));

  decision_sub_ = create_subscription<sea_interfaces::msg::PlanningDecision>(
    "/decision/planning", rclcpp::QoS(10),
    std::bind(&ControlNode::on_decision, this, std::placeholders::_1));

  RCLCPP_INFO(get_logger(), "Control node initialized");
}

// 목표 속도에 따라 가감속을 보정하고 차선 오프셋 기반 조향을 계산합니다.
void ControlNode::on_decision(const sea_interfaces::msg::PlanningDecision::SharedPtr msg)
{
  auto command = sea_interfaces::msg::ControlCommand();
  command.stamp = now();

  const float desired_velocity = msg->target_velocity;
  const float max_acceleration = 1.5F;
  const float max_deceleration = 2.0F;

  if (msg->behavior == "slow_down")
  {
    command.throttle = 0.0F;
    command.brake = std::clamp((max_deceleration + desired_velocity * 0.1F) / max_deceleration, 0.0F, 1.0F);
  }
  else
  {
    command.brake = 0.0F;
    command.throttle = std::clamp(desired_velocity / 20.0F, 0.0F, 1.0F);
  }

  command.steering_angle = std::clamp(msg->target_lane_offset * 0.5F, -0.35F, 0.35F);

  command_pub_->publish(command);
  RCLCPP_DEBUG(get_logger(), "Published control command: throttle %.2f brake %.2f steering %.2f", command.throttle, command.brake, command.steering_angle);
}
}  // namespace control

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<control::ControlNode>());
  rclcpp::shutdown();
  return 0;
}
