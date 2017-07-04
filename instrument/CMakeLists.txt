cmake_minimum_required(VERSION 3.7)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# targets
add_library(instrumentation Pass.cpp)

add_library(llvmCuda SHARED RegisterPass.cpp)
target_link_libraries(llvmCuda instrumentation)
