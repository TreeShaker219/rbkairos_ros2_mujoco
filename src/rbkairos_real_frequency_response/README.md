# RBKairos Real Frequency Response

This package adds an isolated real-robot frequency response test path for the
RBKairos base. It does not modify the existing mentor-owned controller packages.

## Contents

```text
include/rbkairos_real_frequency_response/base_twist_controller.hpp
src/base_twist_controller.cpp
config/base_twist_controller.yaml
launch/base_twist_controller.launch.py
scripts/frequency_response_recorder.py
scripts/analyze_frequency_response.py
scripts/check_real_robot_ready.py
```

The controller plugin is:

```text
rbkairos_real_frequency_response/BaseTwistController
```

The intended controller instance name is:

```text
rbkairos_base_twist_controller
```

## Build

From the ROS2 workspace:

```bash
cd /colcon_ws
colcon build --packages-select rbkairos_real_frequency_response
source install/setup.zsh
```

## Load Controller

Preferred non-invasive loading path:

```bash
ros2 run controller_manager spawner rbkairos_base_twist_controller \
  --controller-manager /controller_manager \
  --controller-type rbkairos_real_frequency_response/BaseTwistController \
  --param-file /colcon_ws/src/rbkairos_real_frequency_response/config/base_twist_controller.yaml \
  --inactive
```

If the local `controller_manager spawner` does not support `--controller-type`
and `--param-file`, do not edit existing controller configs without approval.

Activate the experiment controller only after deactivating `admittance_wbc`:

```bash
ros2 service call /controller_manager/switch_controller controller_manager_msgs/srv/SwitchController \
  "{deactivate_controllers: [admittance_wbc], activate_controllers: [rbkairos_base_twist_controller], strictness: 2}"
```

## Readiness Check

```bash
ros2 run rbkairos_real_frequency_response check_real_robot_ready.py \
  --cmd-topic /base_cmd_vel \
  --odom-topic /rbkairos_base/odom \
  --emergency-stop-topic /emergency_stop \
  --require-cmd-subscriber \
  --require-emergency-stop
```

## Smoke Test

```bash
ros2 run rbkairos_real_frequency_response frequency_response_recorder.py \
  --environment real_smoke \
  --axis x \
  --amplitude 0.005 \
  --frequencies 0.50 \
  --publish-rate 50 \
  --cycles 2 \
  --discard-cycles 0.5 \
  --ramp-cycles 0.5 \
  --cmd-topic /base_cmd_vel \
  --odom-topic /rbkairos_base/odom \
  --require-cmd-subscriber \
  --require-emergency-stop \
  --startup-timeout 10.0 \
  --max-x 0.10 \
  --max-y 0.03 \
  --max-yaw-deg 3 \
  --odom-timeout 0.3 \
  --post-stop-duration 0.5 \
  --output-dir /colcon_ws/src/rbkairos_real_frequency_response/data/frequency_response/real_smoke
```

## Formal Test

The formal test parameters intentionally mirror the MuJoCo baseline frequency
response experiment. Run the smoke test first. The low-frequency points can move
the real robot much farther because `position_amplitude ~= A / (2*pi*f)`.

```bash
ros2 run rbkairos_real_frequency_response frequency_response_recorder.py \
  --environment real_lab \
  --axis x \
  --amplitude 0.10 \
  --frequencies 0.20 0.30 0.50 1.00 1.80 2.80 4.00 \
  --publish-rate 50 \
  --cycles 6 \
  --discard-cycles 2 \
  --ramp-cycles 1.0 \
  --cmd-topic /base_cmd_vel \
  --odom-topic /rbkairos_base/odom \
  --require-cmd-subscriber \
  --require-emergency-stop \
  --startup-timeout 10.0 \
  --max-x 1.50 \
  --max-y 0.50 \
  --max-yaw-deg 180 \
  --odom-timeout 1.0 \
  --post-stop-duration 0.5 \
  --settle-between-frequencies 1.0 \
  --output-dir /colcon_ws/src/rbkairos_real_frequency_response/data/frequency_response/real_lab
```

## Analyze

```bash
ros2 run rbkairos_real_frequency_response analyze_frequency_response.py \
  --experiment-dir /colcon_ws/src/rbkairos_real_frequency_response/data/frequency_response/real_lab/<experiment_dir>
```

Analysis writes:

```text
analysis/frequency_summary.csv
analysis/analysis_metadata.json
analysis/bode_gain_phase.png
analysis/time_series_f_*.png
```

If Matplotlib is unavailable or incompatible with the installed NumPy, the
analyzer still writes `frequency_summary.csv` and skips plots.
