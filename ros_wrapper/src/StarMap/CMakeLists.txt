cmake_minimum_required(VERSION 3.10.2)
project(starmap)

set (CMAKE_CXX_FLAGS_DEBUG
  "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
# set (CMAKE_LINKER_FLAGS_DEBUG
#   "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")


## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin_simple REQUIRED)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system program_options filesystem)

## TODO FIXME do not use the absolute path 
# set(Torch_DIR /home/erl/Workspace/libtorch/share/cmake/Torch)
find_package(Torch 1.4 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(OpenCV 4.2 REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
    roscpp
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}_lib
  CATKIN_DEPENDS roscpp
  DEPENDS Boost
)


catkin_simple(ALL_DEPS_REQUIRED)

## Declare a C++ library
cs_add_library(${PROJECT_NAME}
        src/starmap.cpp
        src/starmap_nodelet.cpp
)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)
target_link_libraries(${PROJECT_NAME}
  "${OpenCV_LIBRARIES}"
  "${TORCH_LIBRARIES}"
)


## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
cs_add_executable(${PROJECT_NAME}_demo src/starmap_demo.cpp)
cs_add_executable(${PROJECT_NAME}_node src/starmap_node.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
set_target_properties(${PROJECT_NAME}_demo PROPERTIES CXX_STANDARD 11)
set_target_properties(${PROJECT_NAME}_node PROPERTIES CXX_STANDARD 11)

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_demo
   ${PROJECT_NAME}
   "${TORCH_LIBRARIES}"
   "${Boost_LIBRARIES}"
   "${OpenCV_LIBRARIES}"
)

target_link_libraries(${PROJECT_NAME}_node
  ${PROJECT_NAME}
  "${TORCH_LIBRARIES}"
  "${Boost_LIBRARIES}"
  "${OpenCV_LIBRARIES}"
  )


cs_install()
cs_export()
#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
catkin_add_gtest(${PROJECT_NAME}-test
        tests/test_crop.cpp
        tests/test_hmparse.cpp
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  "${Boost_LIBRARIES}"
  "${OpenCV_LIBRARIES}"
  gtest_main)

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
