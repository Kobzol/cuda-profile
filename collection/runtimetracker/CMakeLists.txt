cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES RuntimeTracker.cpp)

add_library(runtimetracker SHARED ${SOURCE_FILES} ${PROTOBUF_CXX_FILES})
target_link_libraries(runtimetracker ${CUDA_LIBRARIES})
target_include_directories(runtimetracker PRIVATE ${CUDA_INCLUDE_DIRS})
