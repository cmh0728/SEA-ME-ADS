#!/usr/bin/env python3
"""
Launch lane detection and control nodes together.
Usage: ros2 launch <package> lane_follow.launch.py
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    lane_detector = Node(
        package="lane_detector_py",
        executable="lane_detector_node",
        name="lane_detector",
        output="screen",
    )

    controller = Node(
        package="control",
        executable="control_node",
        name="lane_follow_control",
        output="screen",
    )

    return LaunchDescription([lane_detector, controller])
