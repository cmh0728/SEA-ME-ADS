#include "planning/planning_node.hpp"

#include <memory>

namespace planning
{

PlanningNode::PlanningNode()
: rclcpp::Node("planning_node")
{
  RCLCPP_INFO(get_logger(), "Planning node started");
}

}  // namespace planning

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<planning::PlanningNode>());
  rclcpp::shutdown();
  return 0;
}
