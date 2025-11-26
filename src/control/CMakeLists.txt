cmake_minimum_required(VERSION 3.8)
project(control)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(control_node 
  src/control_node.cpp
  )

add_executable(pid_control
  src/Pid_control.cpp
)
target_include_directories(control_node  PRIVATE include)

target_include_directories(pid_control  PRIVATE include)


ament_target_dependencies(control_node
  rclcpp
  geometry_msgs
  nav_msgs
  std_msgs)

install(TARGETS 
  control_node 
  pid_control
  DESTINATION lib/${PROJECT_NAME})

ament_package()
