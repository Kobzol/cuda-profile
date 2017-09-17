cmake_minimum_required(VERSION 3.8)
project(cuda)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(CUDA REQUIRED)

#set(LLVM_DIR "/home/kobzol/libraries/llvm-5.0.0-build/lib/cmake/llvm")
find_package(LLVM CONFIG REQUIRED)
add_definitions(${LLVM_DEFINITIONS})

message("LLVM found at ${LLVM_DIR}")

find_package(Protobuf REQUIRED)

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
FILE(GLOB PROTOBUF_SRC runtime/protobuf/generated/*.pb.cc)
set(SOURCE_FILES main.cpp kernel.cu general.h ${PROTOBUF_SRC})
cuda_add_executable(cuda ${SOURCE_FILES} OPTIONS "-arch sm_30 -std=c++11")
target_link_libraries(cuda runtime ${PROTOBUF_LIBRARIES})
