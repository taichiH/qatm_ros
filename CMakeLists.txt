cmake_minimum_required(VERSION 2.8.3)
project(qatm_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  cv_bridge
  image_transport
  sensor_msgs
  jsk_recognition_msgs
)

find_package(OpenCV 3 REQUIRED)

set(CMAKE_CXX_FLAGS "--std=c++11")
message("OPENCV ${OpenCV_VERSION} FOUND")

include_directories()

link_directories(${OPENCV_LIBRARY_DIRS})

catkin_package()
