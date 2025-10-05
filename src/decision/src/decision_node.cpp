#include "decision/decision_node.hpp"

#include <algorithm>
#include <functional>
#include <utility>

namespace decision
{
// 판단 노드는 지각 데이터를 받아 감속 여부와 목표 속도를 정합니다.
DecisionNode::DecisionNode()
: rclcpp::Node("decision_node")
{
  decision_pub_ = create_publisher<sea_interfaces::msg::PlanningDecision>(
    "/decision/planning", rclcpp::QoS(10));

  perception_sub_ = create_subscription<sea_interfaces::msg::PerceptionData>(
    "/perception/data", rclcpp::QoS(10),
    std::bind(&DecisionNode::on_perception, this, std::placeholders::_1));

  RCLCPP_INFO(get_logger(), "Decision node initialized");
}

// 안전 거리 기준으로 행동을 선택하고 목표 값을 작성합니다.
void DecisionNode::on_perception(const sea_interfaces::msg::PerceptionData::SharedPtr msg)
{
  auto decision = sea_interfaces::msg::PlanningDecision();
  decision.stamp = now();
  decision.target_lane_offset = -msg->lane_offset;  // try to recenter

  const float safe_distance = 10.0F;
  if (msg->obstacle_distance < safe_distance)
  {
    decision.behavior = "slow_down";
    decision.target_velocity = std::max(0.0F, msg->ego_velocity - 4.0F);
  }
  else
  {
    decision.behavior = "cruise";
    decision.target_velocity = std::min(20.0F, msg->ego_velocity + 1.0F);
  }

  decision_pub_->publish(decision);
  RCLCPP_DEBUG(get_logger(), "Published decision: %s at %.2f m/s", decision.behavior.c_str(), decision.target_velocity);
}
}  // namespace decision

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<decision::DecisionNode>());
  rclcpp::shutdown();
  return 0;
}
