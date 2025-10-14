"""ROS2 helper node that injects automatic driving commands when the dashboard
is switched to AUTO mode.

The node listens to the ``DrivingMode`` channel (via the shared queue
infrastructure) and, once AUTO is selected, publishes example speed and steering
values on the same `SpeedMotor`/`SteerMotor` channels used by the dashboard.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from typing import Mapping, MutableMapping

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

# Allow importing the shared BFMC modules (message handlers, enums, etc.).
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../ros2_ws/src
BRAIN_ROOT = PROJECT_ROOT / "Brain"
if str(BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(BRAIN_ROOT))

from src.utils.messages.allMessages import (  # type: ignore
    DrivingMode,
    SpeedMotor,
    SteerMotor,
)
from src.utils.messages.messageHandlerSender import (  # type: ignore
    messageHandlerSender,
)
from src.utils.messages.messageHandlerSubscriber import (  # type: ignore
    messageHandlerSubscriber,
)


class AutoControlNode(Node):
    """ROS2 node that bridges dashboard AUTO mode to queue-based commands."""

    def __init__(
        self,
        queues_list: Mapping[str, object],
        loop_rate_hz: float = 10.0,
    ) -> None:
        super().__init__("auto_control_node")

        self._queues_list = queues_list
        self._auto_active = False
        self._phase = 0.0

        # Subscribers/Senders over the shared multiprocessing queues.
        self._driving_mode_subscriber = messageHandlerSubscriber(
            self._queues_list, DrivingMode, "lastOnly", True
        )
        self._speed_sender = messageHandlerSender(self._queues_list, SpeedMotor)
        self._steer_sender = messageHandlerSender(self._queues_list, SteerMotor)

        # Timers – poll dashboard state and publish test commands.
        self._poll_timer = self.create_timer(0.1, self._poll_driving_mode)
        self._publish_timer = self.create_timer(
            1.0 / loop_rate_hz, self._publish_auto_commands
        )

    # --------------------------------------------------------------------- internals --
    def _poll_driving_mode(self) -> None:
        mode = self._driving_mode_subscriber.receive()
        if mode is None:
            return

        mode_lower = mode.lower()
        if mode_lower == "auto" and not self._auto_active:
            self.get_logger().info("AUTO mode activated – starting autonomous loop.")
            self._auto_active = True
            self._phase = 0.0
        elif mode_lower != "auto" and self._auto_active:
            self.get_logger().info(f"AUTO mode cancelled by dashboard ({mode_lower}).")
            self._auto_active = False

    def _publish_auto_commands(self) -> None:
        if not self._auto_active:
            return

        # Example profile: constant forward speed + sinusoidal steering sweep.
        speed_mps = 30  # ~4.3 km/h
        steer_deg = 10.0 * math.sin(self._phase)

        scaled_speed = str(int(speed_mps * 10.0))
        scaled_steer = str(int(steer_deg * 10.0))

        self._speed_sender.send(scaled_speed)
        self._steer_sender.send(scaled_steer)

        self.get_logger().debug(
            f"AUTO cmd -> speed={scaled_speed} steer={scaled_steer} (phase={self._phase:.2f})"
        )

        self._phase += 0.1


class _AutoControlRosThread(threading.Thread):
    """Spin a ROS2 node inside a dedicated thread for WorkerProcess integration."""

    def __init__(self, queues_list: Mapping[str, object]) -> None:
        super().__init__()
        self._queues_list = queues_list
        self._stop_event = threading.Event()
        self._executor: SingleThreadedExecutor | None = None
        self._node: AutoControlNode | None = None

    def run(self) -> None:
        rclpy.init(args=None)
        self._node = AutoControlNode(self._queues_list)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        try:
            while not self._stop_event.is_set():
                self._executor.spin_once(timeout_sec=0.1)
        finally:
            if self._executor:
                self._executor.remove_node(self._node)
            if self._node:
                self._node.destroy_node()
            rclpy.shutdown()

    def stop(self) -> None:
        self._stop_event.set()


def create_auto_control_process(queue_list: MutableMapping[str, object]):
    """Factory returning a WorkerProcess-compatible wrapper."""
    from src.templates.workerprocess import WorkerProcess  # type: ignore

    class AutoControlProcess(WorkerProcess):
        def _init_threads(self):
            self.threads.append(_AutoControlRosThread(self.queuesList))

    return AutoControlProcess(queue_list, daemon=True)


# --------------------------------------------------------------------------- demo main --
if __name__ == "__main__":
    from multiprocessing import Queue

    # In production the framework injects a shared queuesList dictionary.
    queue_list: MutableMapping[str, object] = {
        "General": Queue(),
        "Config": Queue(),
    }

    # Minimal dry-run: instantiate the node and send fake AUTO triggers.
    rclpy.init(args=None)
    node = AutoControlNode(queue_list)

    try:
        # Simulate AUTO mode toggling.
        for _ in range(3):
            queue_list["General"].put(
                {
                    "Owner": DrivingMode.Owner.value,
                    "msgID": DrivingMode.msgID.value,
                    "msgType": DrivingMode.msgType.value,
                    "msgValue": "auto",
                }
            )
            # Give the node time to react.
            for _ in range(20):
                rclpy.spin_once(node, timeout_sec=0.1)

            queue_list["General"].put(
                {
                    "Owner": DrivingMode.Owner.value,
                    "msgID": DrivingMode.msgID.value,
                    "msgType": DrivingMode.msgType.value,
                    "msgValue": "manual",
                }
            )
            for _ in range(10):
                rclpy.spin_once(node, timeout_sec=0.1)

            time.sleep(0.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()
