cmake_minimum_required(VERSION 2.8.3)
project(arl_unity_ros)
set(CMAKE_BUILD_TYPE Debug)

set(PKG_DEPENDS
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  rosgraph_msgs    
  tf
  image_transport
  )
set(BUILD_DEPENDS
  ${PKG_DEPENDS}
  )
set(RUN_DEPENDS
  ${PKG_DEPENDS}
  )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")


#######################################
## Find dependencies
#######################################

find_package(catkin REQUIRED ${BUILD_DEPENDS})

find_package(Eigen3)
include_directories(include src/tcpimage ${EIGEN3_INCLUDE_DIR} ${catkin_INCLUDE_DIRS})

#######################################
## Declare Catkin package and installation
#######################################

add_service_files(DIRECTORY srv
  FILES
  Command.srv
  SpawnRobot.srv
  SpawnSensor.srv
  )

add_message_files(DIRECTORY msg
  FILES
  ImageSegmentation.msg
  ImageDetections.msg
  Detection.msg
  JointDescription.msg
  Keypoint.msg
  )
generate_messages(DEPENDENCIES std_msgs geometry_msgs)

if(APPLE)
  catkin_package(
    CATKIN_DEPENDS ${RUN_DEPENDS}
    DEPENDS
    LIBRARIES
    )
else()
  catkin_package(
    CATKIN_DEPENDS ${RUN_DEPENDS}
    DEPENDS
    LIBRARIES socket++
    )
endif()



