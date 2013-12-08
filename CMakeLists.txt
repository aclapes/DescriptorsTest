cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(descriptors_test)

find_package(PCL 1.6 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(OpenCV REQUIRED)

add_executable (descriptors_test descriptors_test.cpp)
target_link_libraries (descriptors_test ${OpenCV_LIBS} ${PCL_LIBRARIES})
