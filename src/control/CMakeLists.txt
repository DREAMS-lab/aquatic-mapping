cmake_minimum_required(VERSION 3.8)
project(control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(px4_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)

  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Install Python scripts
install(PROGRAMS
  scripts/position_controller.py
  scripts/field.py
  scripts/rover_monitor.py
  scripts/temp_query.py
  DESTINATION lib/${PROJECT_NAME}
)

# Files to share
install(DIRECTORY
  launch 
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY
  config 
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY
  urdf 
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
