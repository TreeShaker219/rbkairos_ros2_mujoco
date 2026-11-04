cmake_minimum_required(VERSION 3.8)
project(mujoco_sim_ros2)

set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

# download dependencies
include(FetchContent)

set(MUJOCO_DEP_VERSION_lodepng
    b4ed2cd7ecf61d29076169b49199371456d4f90b
    CACHE STRING "Version of `lodepng` to be fetched."
)

# Fetch lodepng dependency.
if (NOT TARGET lodepng)
  FetchContent_Declare(
      lodepng
      GIT_REPOSITORY https://github.com/lvandeve/lodepng.git
      GIT_TAG ${MUJOCO_DEP_VERSION_lodepng}
  )

  FetchContent_GetProperties(lodepng)
  if (NOT lodepng_POPULATED)
    FetchContent_Populate(lodepng)
    # This is not a CMake project.
    set(LODEPNG_SRCS ${lodepng_SOURCE_DIR}/lodepng.cpp)
    set(LODEPNG_HEADERS ${lodepng_SOURCE_DIR}/lodepng.h)
    add_library(lodepng STATIC ${LODEPNG_HEADERS} ${LODEPNG_SRCS})
    target_compile_options(lodepng PRIVATE ${MUJOCO_MACOS_COMPILE_OPTIONS})
    target_link_options(lodepng PRIVATE ${MUJOCO_MACOS_LINK_OPTIONS})
    target_include_directories(lodepng PUBLIC ${lodepng_SOURCE_DIR})
  endif ()
endif ()

# find dependencies
set(OpenGL_GL_PREFERENCE GLVND)
find_package(ament_cmake REQUIRED)
find_package(mujoco REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Threads REQUIRED)

find_package(rclcpp REQUIRED)
find_package(pluginlib REQUIRED)
find_package(controller_manager REQUIRED)
find_package(ament_index_cpp REQUIRED)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")

add_executable(mujoco_sim
    src/glfw_adapter.cc
    src/glfw_dispatch.cc
    src/physics_step.cpp
    src/platform_ui_adapter.cc
    src/shutdown_signal.cpp
    src/simulate.cc
    src/main.cc)
target_link_libraries(mujoco_sim PUBLIC mujoco::mujoco glfw lodepng Threads::Threads)
ament_target_dependencies(mujoco_sim PUBLIC rclcpp pluginlib controller_manager)
target_include_directories(mujoco_sim PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>" PRIVATE src)

if (BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  add_executable(test_physics_step
      src/physics_step.cpp
      src/shutdown_signal.cpp
      test/test_physics_step.cpp)
  set_target_properties(test_physics_step PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON)
  target_compile_definitions(test_physics_step PRIVATE
      MUJOCO_SIM_MAIN_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc")
  target_link_libraries(test_physics_step
      ${PROJECT_NAME}
      mujoco::mujoco)
  target_include_directories(test_physics_step PRIVATE src)
  ament_target_dependencies(test_physics_step rclcpp)
  add_test(NAME test_physics_step COMMAND test_physics_step)
  set_tests_properties(test_physics_step PROPERTIES TIMEOUT 30)
endif ()

install(TARGETS ${PROJECT_NAME}
    EXPORT export_${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

install(DIRECTORY include/
    DESTINATION include/${PROJECT_NAME})

install(TARGETS mujoco_sim
    DESTINATION lib/${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp pluginlib mujoco)

ament_package()
