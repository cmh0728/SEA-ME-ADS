# SEA:ME Autonomous Driving System

SEA:ME Team 1 autonomous driving stack prototype built on ROS 2 and C++17. The stack is organized around a classic perception→planning→control pipeline to keep responsibilities clear and extensible.

## Workspace Layout

```
SEA-ME-ADS/
├── README.md
└── src/
    ├── control/          # Control stage: converts planning targets to low-level commands
    ├── planning/         # Planning stage: behavior planning based on perception
    ├── perception/       # Perception stage: fuses sensor data and publishes scene state
    ├── sea_bringup/      # Launch files to run the full stack
    └── sea_interfaces/   # Shared message definitions for inter-stage communication
```

Each package is a standard `ament_cmake` ROS 2 package with isolated responsibilities and shared interfaces. The mock nodes provided here simulate the data flow end-to-end so you can iterate on one stage at a time.

## Quick Start

1. Source your ROS 2 setup (e.g. `source /opt/ros/humble/setup.zsh`).
2. Build the workspace from the repository root:
   ```bash
   colcon build --symlink-install
   ```
3. Source the workspace overlay:
   ```bash
   source install/setup.zsh
   ```
4. Launch the full pipeline:
   ```bash
   ros2 launch sea_bringup pipeline.launch.py
   ```

The demo publishes synthetic perception measurements, turns them into planning outputs, and finally generates normalized throttle/brake/steering commands.

## Package Notes

- `sea_interfaces`: Defines the custom ROS messages (`PerceptionData`, `PlanningDecision`, `ControlCommand`) shared across packages.
- `perception`: Periodically publishes mock `PerceptionData`. Replace `publish_mock_measurement()` with real sensor/ML integration.
- `planning`: Subscribes to perception updates, runs a simple rule-based planner, and publishes `PlanningDecision` targets. Extend this node with your behavior planner or trajectory generator.
- `control`: Consumes planning targets and emits `ControlCommand` messages with normalized actuator values. Swap the logic for your MPC/PID controller and integrate with vehicle hardware drivers.
- `sea_bringup`: Launch file that brings the three stages up together. Extend with parameter files, RViz configurations, and additional nodes as the project grows.

## License

- update later
