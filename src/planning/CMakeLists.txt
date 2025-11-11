cmake_minimum_required(VERSION 3.8)
project(planning)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

add_executable(planning_node src/planning_node.cpp)
target_include_directories(planning_node PRIVATE include)
ament_target_dependencies(planning_node rclcpp )

install(TARGETS planning_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
