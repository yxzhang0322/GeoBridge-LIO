cmake_minimum_required(VERSION 2.8.3)
project(fast_lio)
option(ENABLE_RUNTIME_PROFILING "Record buffered per-scan runtime measurements" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

ADD_COMPILE_OPTIONS(-std=c++14 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set( CMAKE_CXX_FLAGS "-std=c++14 -O3" )

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -std=c++0x -std=c++14 -fexceptions")

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

set(FASTLIO_SOURCE_DIR "" CACHE PATH "External FAST-LIO source directory")
if(NOT EXISTS "${FASTLIO_SOURCE_DIR}/include/ikd-Tree/ikd_Tree.cpp" OR
   NOT EXISTS "${FASTLIO_SOURCE_DIR}/include/IKFoM_toolkit/esekfom/esekfom.hpp")
  message(FATAL_ERROR "Set FASTLIO_SOURCE_DIR to an external FAST-LIO checkout with submodules initialized")
endif()

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  rosbag
  std_msgs
  pcl_ros
  tf
  livox_ros_driver
  message_generation
  eigen_conversions
  gnss_comm
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(Ceres REQUIRED)
message(Eigen: ${EIGEN3_INCLUDE_DIR})

include_directories(
    ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${FASTLIO_SOURCE_DIR}/include
  include)

add_message_files(
  FILES
  Pose6D.msg
)

generate_messages(
 DEPENDENCIES
 geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
  DEPENDS EIGEN3 PCL Ceres
  INCLUDE_DIRS
)

add_executable(fastlio_mapping
  src/laserMapping.cpp
  ${FASTLIO_SOURCE_DIR}/include/ikd-Tree/ikd_Tree.cpp
  src/preprocess.cpp
  include/wahba/alignment.cpp
  include/wahba/python_bridge.cpp
  include/wahba/BSpline.cpp
  include/wahba/quasar.cpp
  include/wahba/sdp_solver.cpp
  include/pgo_sw/sliding_window_backend.cpp
  src/WorldMapTransformManager.cpp)
target_link_libraries(fastlio_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES})
target_link_libraries(fastlio_mapping ${CERES_LIBRARIES})
add_dependencies(fastlio_mapping ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_include_directories(fastlio_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})
if(ENABLE_RUNTIME_PROFILING)
  target_compile_definitions(fastlio_mapping PRIVATE ENABLE_RUNTIME_PROFILING)
endif()

add_executable(dataset_frontend_check src/dataset_frontend_check.cpp src/preprocess.cpp)
add_executable(dataset_time_calibration src/dataset_time_calibration.cpp)
target_link_libraries(dataset_time_calibration ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(dataset_frontend_check ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(dataset_frontend_check ${catkin_EXPORTED_TARGETS})
if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(tests/test_dataset_adapters.py)
  catkin_add_gtest(test_tapm_sparse tests/test_tapm_sparse.cpp)
  if(TARGET test_tapm_sparse)
    target_link_libraries(test_tapm_sparse ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})
    add_dependencies(test_tapm_sparse ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})
  endif()
  catkin_add_gtest(test_wahba_vectors tests/test_wahba_vectors.cpp
    include/wahba/alignment.cpp include/wahba/BSpline.cpp include/wahba/python_bridge.cpp)
  if(TARGET test_wahba_vectors)
    target_link_libraries(test_wahba_vectors ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES})
    add_dependencies(test_wahba_vectors ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})
  endif()
endif()

add_executable(rtk_antenna_calibration_tool
  src/rtk_antenna_calibration_tool.cpp)
target_link_libraries(rtk_antenna_calibration_tool ${catkin_LIBRARIES})
