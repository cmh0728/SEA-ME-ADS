from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lane_detector',
            executable='lane_detector_node',
            name='lane_detector',
            output='screen',
            parameters=[{
                'image_topic': '/camera/color/image_raw',   # 실제 카메라 토픽으로 변경
                'publish_overlay_topic': '/lane/overlay',
                'publish_offset_topic': '/lane/center_offset',
                'use_birdeye': True,
                # 실제 캘리브로 바꾸기!
                'src_points': [[200.0, 300.0],[440.0, 300.0],[620.0, 470.0],[20.0, 470.0]],
                'dst_points': [[100.0,   0.0],[540.0,   0.0],[540.0, 480.0],[100.0, 480.0]],
            }],
        )
    ])
