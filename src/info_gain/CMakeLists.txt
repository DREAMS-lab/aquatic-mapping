cmake_minimum_required(VERSION 3.8)
project(info_gain)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(px4_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Install Python scripts
install(PROGRAMS

  scripts/mi_to_px4_relay.py
  scripts/px4_odom_to_tf.py
  scripts/pg_viz.py

  # pure MI
  scripts/1/mi_planner_1.py

  # MI + distance penalty
  scripts/2/mi_planner_2.py


  
  DESTINATION lib/${PROJECT_NAME}
)

# Files to share
install(DIRECTORY
  launch 
  scripts
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
