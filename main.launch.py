#!/usr/bin/env python3
"""
Launch lane detection and control nodes together.
Usage: ros2 launch <package> lane_follow.launch.py
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    CameraNode = Node(
        package="perception",
        executable="CamNode",
        name="CameraProcessing_node",
        output="screen",
    )

    PlanningNode = Node(
        package="planning",
        executable="planning_node",
        name="planning_node",
        output="screen",
    )

    controlNode = Node(
        package="control",
        executable="control_node",
        name="lane_follow_control",
        output="screen",
    )

    return LaunchDescription([CameraNode, PlanningNode ,controlNode])
