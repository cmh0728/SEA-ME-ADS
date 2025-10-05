from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # 지각-판단-제어 노드를 한 번에 실행하는 기본 런치 파일입니다.
    return LaunchDescription(
        [
            Node(
                package="perception",
                executable="perception_node",
                name="perception",
                output="screen",
            ),
            Node(
                package="decision",
                executable="decision_node",
                name="decision",
                output="screen",
            ),
            Node(
                package="control",
                executable="control_node",
                name="control",
                output="screen",
            ),
        ]
    )
