cmake_minimum_required(VERSION 2.8.3)
project(gt_flow)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14 -Wall")

find_package(catkin_simple REQUIRED)
find_package(PCL REQUIRED)

catkin_simple()

cs_add_library(gt_flow_nodelet src/gt_flow_nodelet.cpp)
target_link_libraries(gt_flow_nodelet ${CATKIN_LIBRARIES} ${PCL_LIBRARIES})

cs_install()
install(FILES nodelet_plugin.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

cs_export()
