# RB-KAIROS ROS 2 and MuJoCo Integration

This repository archives the ROS 2 and MuJoCo integration code, experiment scripts and analysis tools used in my bachelor's thesis. The thesis evaluated how closely the simulated RB-KAIROS reproduced the behavior of the real mobile base. Two experiments test the same wheel velocity controller under different loads. A frequency response experiment (FRF) measures how well the freely moving base follows a sinusoidal velocity command. A fixed-tether experiment measures how much steady force the base exerts when a taut cable restrains it.

The RB-KAIROS model data cannot be made public, so all RB-KAIROS model files were removed from this archive, including the robot’s URDF, mesh files and MuJoCo scenes. The launch commands below document how the experiments were run with those files present in the thesis workspace. The original experimental recordings are also excluded from the archive.

## Build the public packages

The original environment used Ubuntu 22.04, ROS 2 Humble, MuJoCo 3.3.0 and Python 3.10. Run the commands below from the workspace root in Bash. Install the ROS dependencies declared in the packages you plan to build. The RB-KAIROS launch also requires the joint state broadcaster, Mecanum drive controller and joint trajectory controller. The C++ build requires a MuJoCo development installation discoverable by CMake, GLFW and OpenGL. If needed, pass `-Dmujoco_DIR=/path/to/cmake/config` through `--cmake-args`, pointing to the directory containing `mujocoConfig.cmake`. The simulator fetches its `lodepng` dependency during configuration.

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select \
  mujoco_sim_ros2 mujoco_ros2_control_msgs mujoco_ros2_control
source install/setup.bash
```

In each additional terminal, source both `/opt/ros/humble/setup.bash` and `install/setup.bash` from the workspace root.

## Launch the RB-KAIROS thesis simulation

The supplied launch file targets a four-wheel Mecanum base with a UR5 arm. It expects the joint names and command interfaces used by the [controller configuration](src/rbkairos/config/mujoco_mecanum_drive.yaml).

In the thesis workspace, the RB-KAIROS model files were located under `src/rbkairos/` before the robot package was built. The default launch expects `urdf/rbkairos.urdf`, `mujoco/rbkairos.xml`, their referenced files and the `meshes/` directory. The traction experiments additionally require `mujoco/rbkairos_scene_tendon_test_x.xml` and `mujoco/rbkairos_scene_tendon_test_y.xml` with the corresponding tendon, force sensor and attachment sites.

```bash
colcon build --symlink-install --packages-select rbkairos
source install/setup.bash
ros2 launch rbkairos mujoco_mecanum_drive.launch.py
```

This opens the free-driving scene used for the FRF experiment. In a second terminal, check that `joint_state_broadcaster`, `mecanum_base_controller` and `ur5_joint_trajectory_controller` are active. Confirm that the simulator publishes base ground-truth odometry.

```bash
ros2 control list_controllers
ros2 topic echo --once --qos-reliability best_effort /base_ground_truth_odom
```

The examples below assume the default asset layout. For experiments, use the same UR5 `up` initial pose, wheel contact settings and controller parameters across matched runs. The launch holds the arm at the initial joint positions specified in the URDF. The Python recorder and analysis environment needs the ROS packages and the Python dependencies in [pyproject.toml](pyproject.toml). Run `uv sync --locked` and activate `.venv` in the recorder terminal with the ROS environment sourced. Check that `python3 -c 'import rclpy, mujoco, numpy, matplotlib'` succeeds.

## Measure velocity frequency response

With the free-driving scene running in terminal A, record one frequency in terminal B. This example sends a 0.20 Hz sinusoid along X at a velocity amplitude of 0.10 m/s. The recorder logs the command it actually publishes and the base velocity measured on `/base_ground_truth_odom`.

```bash
ros2 run rbkairos frequency_response_recorder.py \
  --axis x --amplitude 0.10 --frequencies 0.20 --publish-rate 200 \
  --cycles 6 --ramp-cycles 1 --discard-cycles 2 --max-x 1.5 \
  --output-dir data/frf/candidate_01
```

For the thesis curve, restart the free-driving simulation for each remaining frequency (0.30, 0.50, 1.00, 1.80, 2.80 and 4.00 Hz) and change only `--frequencies`. Restarting resets the model and controller integral state. Keep the same output parent for these seven runs and use a new parent for each controller setting.

```bash
ros2 run rbkairos analyze_frequency_response_fft.py \
  --experiment-dir data/frf/candidate_01 --output-source odom_twist_vx \
  --grid-rate-hz 100 --analysis-cycles 4
```

The analyzer reads `cmd_vel.csv` and `odom.csv`, skips the first two cycles and resamples the common four-cycle window. At the command frequency, gain in dB is `20 log10(output amplitude / input amplitude)`. In `data/frf/candidate_01/fft_analysis/frequency_fft_summary.csv`, check `valid` and `reject_reason` before plotting `fft_gain_db` against `f_hz`. Compare each point with the mean real-robot gain at the same frequency and command amplitude. The same directory contains the HTML report with phase and harmonic diagnostics.

## Measure fixed-tether traction

Stop the free-driving simulation. For the first traction condition, launch the X scene in terminal A. Its spatial tendon links the robot to a fixed world anchor. The observer reads the MuJoCo tendon limit force as the base drives forward against the cable.

```bash
ros2 launch rbkairos mujoco_mecanum_drive.launch.py \
  model_file:=mujoco/rbkairos_scene_tendon_test_x.xml traction_tendon_enabled:=true
```

Once the controllers are active, record a pull at 0.01 m/s in terminal B. The recorder checks the model, sensor, arm posture and initial cable tension before applying the command.

```bash
ros2 run rbkairos traction_tendon_experiment.py \
  --axis x --velocity-x 0.01 --expected-ur5-preset up \
  --output-dir data/traction/candidate_01
```

Repeat with a fresh simulation for each condition. The full set uses 0.01, 0.05 and 0.10 m/s in **each** of X and Y. For Y, use the launch below and change the recorder arguments to `--axis y --velocity-y SPEED`, substituting the target speed in m/s.

```bash
ros2 launch rbkairos mujoco_mecanum_drive.launch.py \
  model_file:=mujoco/rbkairos_scene_tendon_test_y.xml traction_tendon_enabled:=true \
  traction_tendon_name:=traction_cable_y traction_tendon_sensor_name:=traction_cable_y_limit_force \
  traction_tendon_attach_site_name:=traction_attach_y traction_tendon_anchor_site_name:=traction_anchor_y
```

Each timestamped run contains the sampled tension in `traction_tendon_state.csv` and a `traction_tendon_summary.json`. To reanalyze all runs for this setting together, use the candidate directory as the batch input.

```bash
ros2 run rbkairos analyze_traction_tendon.py data/traction/candidate_01 --batch \
  --output-dir data/traction/candidate_01/analysis
```

Read `traction_tendon_batch_summary.csv` in the analysis directory. Use the `run_dir` column to look up `command_profile.target_axis_velocity_mps` in each run’s `run_metadata.json`. For valid runs, plot `steady_tow_force_n` against that target speed separately for X and Y. This value is the median active cable tension over the accepted steady interval. Compare it with the real tether reading for the same axis and speed. A startup peak or a run that timed out before reaching steady state cannot supply that comparison point.

## Compare controller settings

Hold two of `Kp`, `Ki` and `i_max` fixed and change the third. Compare how the seven-point gain curve and the two three-point traction curves move relative to the real measurements. A setting can follow the velocity command more closely at higher frequencies yet overpredict steady pulling force in one direction. The thesis uses these differences to explain parameter choices.

If you already have recordings, you can run both analyzers directly with Python from the workspace root, using the arguments shown above. For example, run `python3 src/rbkairos/scripts/analyze_traction_tendon.py data/traction/candidate_01 --batch`. Offline analysis of these recordings does not require the laboratory models.

For automated runs, the [evaluation configuration](src/rbkairos/config/evaluation/default.yaml) defines the frequencies, six traction conditions and reference inputs. The [evaluation tool](src/rbkairos/scripts/evaluate_parameters.py) can launch experiments in managed mode or combine existing runs in offline mode. Supply the real FRF reference CSV separately when running a reference comparison. The [real-robot FRF guide](src/rbkairos_real_frequency_response/README.md) describes how the lab recordings were collected.

## Upstream sources and local changes

The upstream project for `mujoco_ros2_control` is [moveit/mujoco_ros2_control](https://github.com/moveit/mujoco_ros2_control). This workspace imported the `humble` branch of the [rxdu fork](https://github.com/rxdu/mujoco_ros2_control) and added the local changes described below.

| Local directory | Imported from | Recorded imported revision |
| --- | --- | --- |
| `src/mujoco_ros2_control` | [rxdu/mujoco_ros2_control](https://github.com/rxdu/mujoco_ros2_control) | `humble`, `7af46c09c740f2566ade6e8654c98166443c951a` |
| `src/mujoco_sim_ros2` | [rxdu/mujoco_sim_ros2](https://github.com/rxdu/mujoco_sim_ros2) | `devel`, `01e2a14efa52735245b5d3f5d7967c70bf9f6b29` |

Local changes include post-step physics plugin callbacks, joint and contact diagnostic messages, velocity PID diagnostics and effort limiting, and experiment automation. The integration publishes joint and contact diagnostics on `/mujoco_dof_state` and `/mujoco_contact_state`.

| Package | Purpose |
| --- | --- |
| `mujoco_sim_ros2` | MuJoCo simulation application with a ROS 2 physics plugin interface. |
| `mujoco_ros2_control` | Integration of MuJoCo joints and sensors with `ros2_control`, including PID diagnostics and effort limits. |
| `mujoco_ros2_control_msgs` | Messages for MuJoCo degrees of freedom, contact forces, and controller diagnostics. |
| `rbkairos` | Robot-specific plugins, controller configuration, experiment recorders, offline analysis, and parameter evaluation. |
| `rbkairos_real_frequency_response` | Real-robot frequency response controller, recorder, and analysis tools. |
