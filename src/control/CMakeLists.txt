cmake_minimum_required(VERSION 3.8)
project(control)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sea_interfaces REQUIRED)

add_executable(control_node src/control_node.cpp)
target_include_directories(control_node PRIVATE include)
ament_target_dependencies(control_node rclcpp sea_interfaces)

install(TARGETS control_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
