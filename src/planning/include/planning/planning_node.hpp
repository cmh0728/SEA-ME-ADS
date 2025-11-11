#ifndef PLANNING__PLANNING_NODE_HPP_
#define PLANNING__PLANNING_NODE_HPP_

#include "rclcpp/rclcpp.hpp"

namespace planning
{

class PlanningNode : public rclcpp::Node
{
public:
  PlanningNode();
};

}  // namespace planning

#endif  // PLANNING__PLANNING_NODE_HPP_
