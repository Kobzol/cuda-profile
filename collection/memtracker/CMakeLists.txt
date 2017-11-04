cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES memtracker.cpp)

add_library(memtracker SHARED ${SOURCE_FILES} ${PROTOBUF_CXX_FILES})
target_link_libraries(memtracker ${CUDA_LIBRARIES})
target_include_directories(memtracker PRIVATE ${CUDA_INCLUDE_DIRS})
