"""ROS2 bridge that forwards /cmd_vel commands to the BFMC serial pipeline.

When AUTO mode is activated on the dashboard, this node reads geometry_msgs/Twist
messages from `/cmd_vel` and converts them to the SpeedMotor/SteerMotor queue
updates consumed by `threadWrite`.
"""

from __future__ import annotations

import math
import sys
import threading
from pathlib import Path
from typing import Mapping, MutableMapping

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

# Enable imports of BFMC frameworks.
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../ros2_ws/src/Brain
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.messages.allMessages import DrivingMode, SpeedMotor, SteerMotor  # type: ignore
from src.utils.messages.messageHandlerSender import messageHandlerSender  # type: ignore
from src.utils.messages.messageHandlerSubscriber import (  # type: ignore
    messageHandlerSubscriber,
)


class CmdVelBridgeNode(Node):
    """Forward `/cmd_vel` to the serial handler when AUTO mode is active."""

    def __init__(self, queues_list: Mapping[str, object]) -> None:
        super().__init__("cmd_vel_bridge")
        self._queues_list = queues_list
        self._auto_active = False

        self._driving_mode_subscriber = messageHandlerSubscriber(
            self._queues_list, DrivingMode, "lastOnly", True
        )

        self._speed_sender = messageHandlerSender(self._queues_list, SpeedMotor)
        self._steer_sender = messageHandlerSender(self._queues_list, SteerMotor)

        self._cmd_vel_sub = self.create_subscription(
            Twist, "/cmd_vel", self._handle_cmd_vel, 10
        )

        self._mode_poll_timer = self.create_timer(0.1, self._poll_driving_mode)

    # ------------------------------------------------------------------ callbacks --
    def _poll_driving_mode(self) -> None:
        mode = self._driving_mode_subscriber.receive()
        if mode is None:
            return

        mode_lower = mode.lower()
        if mode_lower == "auto" and not self._auto_active:
            self.get_logger().info("AUTO mode activated – /cmd_vel bridge enabled.")
            self._auto_active = True
        elif mode_lower != "auto" and self._auto_active:
            self.get_logger().info(f"AUTO mode exit ({mode_lower}); bridge paused.")
            self._auto_active = False

    def _handle_cmd_vel(self, msg: Twist) -> None:
        if not self._auto_active:
            return

        speed_mps = msg.linear.x
        steer_deg = math.degrees(msg.angular.z)

        scaled_speed = str(int(speed_mps * 10.0))
        scaled_steer = str(int(steer_deg * 10.0))

        self._speed_sender.send(scaled_speed)
        self._steer_sender.send(scaled_steer)

        self.get_logger().debug(
            f"/cmd_vel -> speed={scaled_speed} steer={scaled_steer}"
        )


class _CmdVelBridgeThread(threading.Thread):
    """Spin the ROS2 node in a separate thread for WorkerProcess integration."""

    def __init__(self, queues_list: Mapping[str, object]) -> None:
        super().__init__()
        self._queues_list = queues_list
        self._stop_event = threading.Event()
        self._executor: SingleThreadedExecutor | None = None
        self._node: CmdVelBridgeNode | None = None

    def run(self) -> None:
        rclpy.init(args=None)
        self._node = CmdVelBridgeNode(self._queues_list)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        try:
            while not self._stop_event.is_set():
                self._executor.spin_once(timeout_sec=0.1)
        finally:
            if self._executor and self._node:
                self._executor.remove_node(self._node)
            if self._node:
                self._node.destroy_node()
            rclpy.shutdown()

    def stop(self) -> None:
        self._stop_event.set()


def create_cmd_vel_bridge_process(queue_list: MutableMapping[str, object]):
    """Factory compatible with WorkerProcess usage in main.py."""
    from src.templates.workerprocess import WorkerProcess  # type: ignore

    class CmdVelBridgeProcess(WorkerProcess):
        def _init_threads(self):
            self.threads.append(_CmdVelBridgeThread(self.queuesList))

    return CmdVelBridgeProcess(queue_list, daemon=True)


if __name__ == "__main__":
    from multiprocessing import Queue
    import time

    queue_list: MutableMapping[str, object] = {
        "General": Queue(),
        "Config": Queue(),
    }

    rclpy.init(args=None)
    node = CmdVelBridgeNode(queue_list)

    try:
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()



# ex) ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.1}}" -r 5