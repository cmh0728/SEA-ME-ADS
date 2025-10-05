cmake_minimum_required(VERSION 3.8)
project(perception)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sea_interfaces REQUIRED)

add_executable(perception_node src/perception_node.cpp)
target_include_directories(perception_node PRIVATE include)
ament_target_dependencies(perception_node rclcpp sea_interfaces)

install(TARGETS perception_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
