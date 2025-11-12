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
find_package(Eigen3 REQUIRED)

add_executable(CameraProcessing src/CamNode.cpp)
target_include_directories(CameraProcessing
  PRIVATE
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(CameraProcessing rclcpp sensor_msgs cv_bridge)
target_link_libraries(CameraProcessing Eigen3::Eigen ${OpenCV_LIBRARIES})


install(TARGETS CameraProcessing
  DESTINATION lib/${PROJECT_NAME})

ament_export_dependencies(rclcpp sensor_msgs cv_bridge OpenCV)

ament_package()
