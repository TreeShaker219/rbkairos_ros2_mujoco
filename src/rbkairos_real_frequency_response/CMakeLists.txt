cmake_minimum_required(VERSION 3.8)
project(rbkairos_real_frequency_response)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(PKG_DEPS
  controller_interface
  geometry_msgs
  hardware_interface
  nav_msgs
  pluginlib
  rclcpp
  rclcpp_lifecycle
  realtime_tools
  std_msgs
)

find_package(ament_cmake REQUIRED)
foreach(Dependency IN ITEMS ${PKG_DEPS})
  find_package(${Dependency} REQUIRED)
endforeach()

add_library(base_twist_controller SHARED src/base_twist_controller.cpp)
target_compile_features(base_twist_controller PUBLIC cxx_std_17)
target_include_directories(base_twist_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(base_twist_controller PUBLIC ${PKG_DEPS})

pluginlib_export_plugin_description_file(
  controller_interface rbkairos_real_frequency_response.xml
)

install(
  TARGETS base_twist_controller
  EXPORT export_rbkairos_real_frequency_response
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/frequency_response_recorder.py
    scripts/analyze_frequency_response.py
    scripts/check_real_robot_ready.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_targets(export_rbkairos_real_frequency_response HAS_LIBRARY_TARGET)
ament_export_dependencies(${PKG_DEPS})
ament_package()
