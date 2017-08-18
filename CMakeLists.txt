cmake_minimum_required(VERSION 3.8)
project(cuda)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(CUDA)

find_package(LLVM CONFIG)
add_definitions(${LLVM_DEFINITIONS})

set(CMAKE_CXX_COMPILER "${LLVM_TOOLS_BINARY_DIR}/clang++")
set(CUDA_HOST_COMPILER "g++")

# Instrumentation
add_subdirectory("instrument")

# Runtime
add_subdirectory("runtime")

# Tests
add_custom_target(tests COMMAND py.test ../tests)
add_dependencies(tests instrument)

# CUDA program
set(SOURCE_FILES main.cpp kernel.cu general.h)
cuda_add_executable(cuda ${SOURCE_FILES} OPTIONS "-arch sm_30 -std=c++11")
target_link_libraries(cuda runtime)
