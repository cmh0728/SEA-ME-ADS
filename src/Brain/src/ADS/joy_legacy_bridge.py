"""ROS2 bridge mapping `/joy` inputs to BFMC legacy (remote) driving commands.

When the dashboard switches to LEGACY mode, this node converts joystick values
to SpeedMotor / SteerMotor queue updates consumed by `threadWrite`.
"""

from __future__ import annotations

import math
import sys
import threading
from pathlib import Path
from typing import Mapping, MutableMapping

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Joy

# Enable imports of BFMC frameworks.
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../ros2_ws/src/Brain
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.messages.allMessages import DrivingMode, SpeedMotor, SteerMotor  # type: ignore
from src.utils.messages.messageHandlerSender import messageHandlerSender  # type: ignore
from src.utils.messages.messageHandlerSubscriber import (  # type: ignore
    messageHandlerSubscriber,
)


class JoyLegacyBridgeNode(Node):
    """Forward `/joy` axes to the serial handler when LEGACY mode is active."""

    def __init__(self, queues_list: Mapping[str, object]) -> None:
        super().__init__("joy_legacy_bridge")
        self._queues_list = queues_list
        self._legacy_active = False
        self._last_speed: int | None = None
        self._last_steer: int | None = None

        # ROS parameters to allow quick tuning without code changes.
        self.declare_parameter("speed_axis", 1)
        self.declare_parameter("steer_axis", 3)
        self.declare_parameter("speed_scale", 500.0)
        self.declare_parameter("steer_scale", 250.0)
        self.declare_parameter("speed_deadzone", 0.05)
        self.declare_parameter("steer_deadzone", 0.05)
        self.declare_parameter("speed_invert", False)
        self.declare_parameter("steer_invert", True)

        self._speed_axis = int(self.get_parameter("speed_axis").value)
        self._steer_axis = int(self.get_parameter("steer_axis").value)
        self._speed_scale = float(self.get_parameter("speed_scale").value)
        self._steer_scale = float(self.get_parameter("steer_scale").value)
        self._speed_deadzone = float(self.get_parameter("speed_deadzone").value)
        self._steer_deadzone = float(self.get_parameter("steer_deadzone").value)
        self._speed_invert = bool(self.get_parameter("speed_invert").value)
        self._steer_invert = bool(self.get_parameter("steer_invert").value)

        self._driving_mode_subscriber = messageHandlerSubscriber(
            self._queues_list, DrivingMode, "lastOnly", True
        )
        self._speed_sender = messageHandlerSender(self._queues_list, SpeedMotor)
        self._steer_sender = messageHandlerSender(self._queues_list, SteerMotor)

        self._joy_sub = self.create_subscription(Joy, "/joy", self._handle_joy, 10)
        self._mode_poll_timer = self.create_timer(0.1, self._poll_driving_mode)

    # ------------------------------------------------------------------ callbacks --
    def _poll_driving_mode(self) -> None:
        mode = self._driving_mode_subscriber.receive()
        if mode is None:
            return

        mode_lower = mode.lower()
        if mode_lower == "legacy" and not self._legacy_active:
            self.get_logger().info("LEGACY mode activated – joystick bridge enabled.")
            self._legacy_active = True
            self._last_speed = None
            self._last_steer = None
        elif mode_lower != "legacy" and self._legacy_active:
            self.get_logger().info(f"LEGACY mode exit ({mode_lower}); bridge paused.")
            self._legacy_active = False
            self._send_zero_commands()

    def _handle_joy(self, msg: Joy) -> None:
        if not self._legacy_active:
            return

        axes = list(msg.axes)

        speed_value = self._extract_axis(
            axes, self._speed_axis, self._speed_deadzone, self._speed_invert
        )
        steer_value = self._extract_axis(
            axes, self._steer_axis, self._steer_deadzone, self._steer_invert
        )

        if speed_value is None or steer_value is None:
            return

        speed_cmd = int(round(speed_value * self._speed_scale))
        steer_cmd = int(round(steer_value * self._steer_scale))

        # Avoid flooding queues with identical values.
        if self._last_speed != speed_cmd:
            self._speed_sender.send(str(speed_cmd))
            self._last_speed = speed_cmd
            self.get_logger().debug(f"/joy -> speed={speed_cmd}")

        if self._last_steer != steer_cmd:
            self._steer_sender.send(str(steer_cmd))
            self._last_steer = steer_cmd
            self.get_logger().debug(f"/joy -> steer={steer_cmd}")

    # ---------------------------------------------------------------- helpers --
    def _extract_axis(
        self,
        axes: list[float],
        index: int,
        deadzone: float,
        invert: bool,
    ) -> float | None:
        if not 0 <= index < len(axes):
            self.get_logger().warn(
                f"Configured axis index {index} out of range for incoming Joy message."
            )
            return None

        value = axes[index]
        if invert:
            value = -value

        if math.fabs(value) < deadzone:
            value = 0.0

        # Clamp to [-1.0, 1.0] to guard against noisy controllers.
        return max(-1.0, min(1.0, value))

    def _send_zero_commands(self) -> None:
        if self._last_speed not in (None, 0):
            self._speed_sender.send("0")
        if self._last_steer not in (None, 0):
            self._steer_sender.send("0")
        self._last_speed = 0
        self._last_steer = 0


class _JoyLegacyBridgeThread(threading.Thread):
    """Spin the ROS2 node in a separate thread for WorkerProcess integration."""

    def __init__(self, queues_list: Mapping[str, object]) -> None:
        super().__init__()
        self._queues_list = queues_list
        self._stop_event = threading.Event()
        self._executor: SingleThreadedExecutor | None = None
        self._node: JoyLegacyBridgeNode | None = None

    def run(self) -> None:
        rclpy.init(args=None)
        self._node = JoyLegacyBridgeNode(self._queues_list)
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


def create_joy_legacy_bridge_process(queue_list: MutableMapping[str, object]):
    """Factory compatible with WorkerProcess usage in main.py."""
    from src.templates.workerprocess import WorkerProcess  # type: ignore

    class JoyLegacyBridgeProcess(WorkerProcess):
        def _init_threads(self):
            self.threads.append(_JoyLegacyBridgeThread(self.queuesList))

    return JoyLegacyBridgeProcess(queue_list, daemon=True)


if __name__ == "__main__":
    from multiprocessing import Queue
    import time

    queue_list: MutableMapping[str, object] = {
        "General": Queue(),
        "Config": Queue(),
    }

    rclpy.init(args=None)
    node = JoyLegacyBridgeNode(queue_list)

    try:
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
