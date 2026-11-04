cmake_minimum_required(VERSION 3.8)
project(rbkairos)

cmake_policy(SET CMP0072 NEW)
set(OpenGL_GL_PREFERENCE GLVND)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(mujoco REQUIRED)
find_package(mujoco_ros2_control_msgs REQUIRED)
find_package(mujoco_sim_ros2 REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

set(THIS_PACKAGE_DEPENDS
  geometry_msgs
  mujoco_ros2_control_msgs
  mujoco_sim_ros2
  nav_msgs
  pluginlib
  rclcpp
  std_msgs
)

add_library(rbkairos_base_ground_truth_plugin SHARED
  src/base_ground_truth_plugin.cpp
)
target_include_directories(rbkairos_base_ground_truth_plugin
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(rbkairos_base_ground_truth_plugin
  ${THIS_PACKAGE_DEPENDS}
)
target_link_libraries(rbkairos_base_ground_truth_plugin
  mujoco::mujoco
)

add_library(rbkairos_traction_tendon_observer_plugin SHARED
  src/traction_tendon_observer_plugin.cpp
)
target_include_directories(rbkairos_traction_tendon_observer_plugin
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(rbkairos_traction_tendon_observer_plugin
  ${THIS_PACKAGE_DEPENDS}
)
target_link_libraries(rbkairos_traction_tendon_observer_plugin
  mujoco::mujoco
)
pluginlib_export_plugin_description_file(mujoco_sim_ros2 rbkairos_mujoco_plugins.xml)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  ament_add_pytest_test(test_traction_tendon_model
    test/test_traction_tendon_model.py
    TIMEOUT 60
  )
  ament_add_pytest_test(test_traction_tendon_observer_contract
    test/test_traction_tendon_observer_contract.py
    TIMEOUT 30
  )
  ament_add_pytest_test(test_traction_tendon_launch
    test/test_traction_tendon_launch.py
    TIMEOUT 30
  )
  ament_add_pytest_test(test_traction_tendon_analysis
    test/test_traction_tendon_analysis.py
    TIMEOUT 60
  )
  ament_add_pytest_test(test_traction_tendon_recorder
    test/test_traction_tendon_recorder.py
    TIMEOUT 30
  )
  ament_add_pytest_test(test_velocity_pid_diagnostics_contract
    test/test_velocity_pid_diagnostics_contract.py
    TIMEOUT 30
  )
  ament_add_pytest_test(test_traction_tendon_i_max_pilot
    test/test_traction_tendon_i_max_pilot.py
    TIMEOUT 60
  )
  ament_add_pytest_test(test_traction_tendon_steady
    test/test_traction_tendon_steady.py
    TIMEOUT 60
  )
  ament_add_pytest_test(test_traction_evaluator
    test/test_traction_evaluator.py
    TIMEOUT 30
  )
  ament_add_pytest_test(test_traction_metadata_validation
    test/test_traction_metadata_validation.py
    TIMEOUT 30
  )
  ament_add_pytest_test(test_traction_runtime_assets
    test/test_traction_runtime_assets.py
    TIMEOUT 60
  )
  ament_add_pytest_test(test_traction_runners
    test/test_traction_runners.py
    TIMEOUT 30
  )
  ament_add_pytest_test(test_evaluation_result_schema
    test/test_evaluation_result_schema.py
    TIMEOUT 30
  )
  ament_add_pytest_test(test_evaluation_i_max_multispeed
    test/test_evaluation_i_max_multispeed.py
    TIMEOUT 30
  )
  ament_add_pytest_test(test_traction_candidate_outputs
    test/test_traction_candidate_outputs.py
    TIMEOUT 60
  )
  ament_add_pytest_test(test_evaluation_campaign
    test/test_evaluation_campaign.py
    TIMEOUT 30
  )
  ament_add_pytest_test(test_frf_regression
    test/test_frf_regression.py
    TIMEOUT 30
  )
  ament_add_pytest_test(test_frf_i_max_sensitivity
    test/test_frf_i_max_sensitivity.py
    TIMEOUT 60
  )
  ament_add_pytest_test(test_frf_conditional_reuse
    test/test_frf_conditional_reuse.py
    TIMEOUT 30
  )
  ament_add_pytest_test(test_evaluation_campaign_executor
    test/test_evaluation_campaign_executor.py
    TIMEOUT 30
  )
  ament_add_pytest_test(test_removed_load_path
    test/test_removed_load_path.py
    TIMEOUT 30
  )

  add_executable(test_traction_tendon_model_cpp
    test/test_traction_tendon_model_cpp.cpp
  )
  target_compile_definitions(test_traction_tendon_model_cpp
    PRIVATE RBKAIROS_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/mujoco"
  )
  set_target_properties(test_traction_tendon_model_cpp PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
  )
  target_link_libraries(test_traction_tendon_model_cpp
    mujoco::mujoco
  )
  add_test(
    NAME test_traction_tendon_model_cpp
    COMMAND test_traction_tendon_model_cpp
  )
  set_tests_properties(test_traction_tendon_model_cpp PROPERTIES TIMEOUT 60)
endif()

install(
DIRECTORY launch urdf meshes config mujoco scripts
DESTINATION share/${PROJECT_NAME}
PATTERN "__pycache__" EXCLUDE
PATTERN "*.pyc" EXCLUDE)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS
    rbkairos_base_ground_truth_plugin
    rbkairos_traction_tendon_observer_plugin
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# 安装平滑器脚本为可执行，便于 ros2 run rbkairos cmd_vel_smoother
install(PROGRAMS scripts/cmd_vel_smoother.py
  DESTINATION lib/${PROJECT_NAME})
# nav2-style 加速度限幅版本
install(PROGRAMS scripts/cmd_vel_smoother_nav2_style.py
  DESTINATION lib/${PROJECT_NAME})
# Forward/backward motion test recorder
install(PROGRAMS scripts/back_and_forth_motion_recorder.py
  DESTINATION lib/${PROJECT_NAME})
# UR5 arm preset trajectory commander
install(PROGRAMS scripts/arm_preset_commander.py
  DESTINATION lib/${PROJECT_NAME})
# Closed-loop position tracking validation experiment
install(PROGRAMS scripts/position_tracking_validation.py
  DESTINATION lib/${PROJECT_NAME})
# Closed-loop base velocity frequency-response experiment
install(PROGRAMS scripts/frequency_response_recorder.py
  DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/analyze_frequency_response.py
  DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/analyze_frequency_response_fft.py
  DESTINATION lib/${PROJECT_NAME})
# Traction-tendon experiment and shared offline analyzer (C2).
install(PROGRAMS scripts/traction_tendon_experiment.py
  DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/analyze_traction_tendon.py
  DESTINATION lib/${PROJECT_NAME})
install(FILES scripts/traction_tendon_analysis.py
  DESTINATION lib/${PROJECT_NAME})
install(FILES scripts/traction_tendon_steady.py
  DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/run_traction_tendon_i_max_pilot.py
  DESTINATION lib/${PROJECT_NAME})
install(FILES scripts/traction_tendon_i_max_pilot.py
  DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/run_frf_i_max_sensitivity_pilot.py
  DESTINATION lib/${PROJECT_NAME})
install(FILES scripts/frf_i_max_sensitivity.py
  DESTINATION lib/${PROJECT_NAME})
# Parameter optimization evaluation entry point
install(PROGRAMS scripts/evaluate_parameters.py
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY evaluation
  DESTINATION lib/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE)
install(DIRECTORY config/evaluation
  DESTINATION lib/${PROJECT_NAME}/config)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(${THIS_PACKAGE_DEPENDS})

ament_package()
