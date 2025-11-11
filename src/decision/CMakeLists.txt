cmake_minimum_required(VERSION 3.8)
project(decision)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

add_executable(decision_node src/decision_node.cpp)
target_include_directories(decision_node PRIVATE include)
ament_target_dependencies(decision_node rclcpp )

install(TARGETS decision_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
