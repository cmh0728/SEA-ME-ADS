#ifndef DECISION__DECISION_NODE_HPP_
#define DECISION__DECISION_NODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "sea_interfaces/msg/perception_data.hpp"
#include "sea_interfaces/msg/planning_decision.hpp"

namespace decision
{
// 지각 정보를 구독해 단순 규칙 기반 행동 결정을 수행하는 노드입니다.
class DecisionNode : public rclcpp::Node
{
public:
  DecisionNode();

private:
  void on_perception(const sea_interfaces::msg::PerceptionData::SharedPtr msg);

  rclcpp::Subscription<sea_interfaces::msg::PerceptionData>::SharedPtr perception_sub_;
  rclcpp::Publisher<sea_interfaces::msg::PlanningDecision>::SharedPtr decision_pub_;
};
}  // namespace decision

#endif  // DECISION__DECISION_NODE_HPP_
