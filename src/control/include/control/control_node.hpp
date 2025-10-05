#ifndef CONTROL__CONTROL_NODE_HPP_
#define CONTROL__CONTROL_NODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "sea_interfaces/msg/planning_decision.hpp"
#include "sea_interfaces/msg/control_command.hpp"

namespace control
{
// 계획 단계의 출력으로부터 차량 제어 명령을 계산하는 노드입니다.
class ControlNode : public rclcpp::Node
{
public:
  ControlNode();

private:
  void on_decision(const sea_interfaces::msg::PlanningDecision::SharedPtr msg);

  rclcpp::Subscription<sea_interfaces::msg::PlanningDecision>::SharedPtr decision_sub_;
  rclcpp::Publisher<sea_interfaces::msg::ControlCommand>::SharedPtr command_pub_;
};
}  // namespace control

#endif  // CONTROL__CONTROL_NODE_HPP_
