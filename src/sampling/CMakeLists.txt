cmake_minimum_required(VERSION 3.8)
project(sampling)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_ros REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Install Python scripts (rover)
install(PROGRAMS
  scripts/rover/rover_monitor.py
  scripts/rover/position_controller.py
  scripts/rover/temp_query.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python scripts (fields)
install(PROGRAMS
  scripts/fields/radial_field.py
  scripts/fields/x_compress_field.py
  scripts/fields/y_compress_field.py
  scripts/fields/x_compress_tilt_field.py
  scripts/fields/y_compress_tilt_field.py
  scripts/fields/world_frame_publisher.py
  scripts/fields/field_filter.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python scripts (missions)
install(PROGRAMS
  scripts/missions/lawnmower.py
  scripts/missions/waypoint_runner.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python scripts (logging)
install(PROGRAMS
  scripts/logging/record_field_data.py
  scripts/logging/logger.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python scripts (analysis)
install(PROGRAMS
  scripts/analysis/gp.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install directories
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

install(DIRECTORY
  data
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
