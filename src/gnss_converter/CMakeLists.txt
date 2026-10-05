cmake_minimum_required(VERSION 3.0.2)
project(gnss_converter)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  gnss_comm # 确保这个包是可找到的
  roslib      # <--- 添加 roslib
  tf2
  tf2_ros
)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(pvt_to_evo_node src/pvt_to_evo.cpp)

target_link_libraries(pvt_to_evo_node
  ${catkin_LIBRARIES}
)
