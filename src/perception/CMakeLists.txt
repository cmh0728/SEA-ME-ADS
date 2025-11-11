cmake_minimum_required(VERSION 3.8)
project(perception)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(perception_node src/perception_node.cpp)
target_include_directories(perception_node
  PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(perception_node rclcpp sensor_msgs cv_bridge)
target_link_libraries(perception_node ${OpenCV_LIBRARIES})

install(TARGETS perception_node
  DESTINATION lib/${PROJECT_NAME})

ament_export_dependencies(rclcpp sensor_msgs cv_bridge OpenCV)

ament_package()
