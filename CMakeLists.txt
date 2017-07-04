cmake_minimum_required(VERSION 3.7)
project(cuda)

set(CMAKE_VERBOSE_MAKEFILE ON)
file(GLOB CUDA_FILES "*.cu")
set_source_files_properties(${CUDA_FILES} PROPERTIES LANGUAGE CXX)

# LLVM
set(LLVM_DIR "/home/kobzol/libraries/llvm-4.0.0")
include_directories(${LLVM_DIR}/include)
link_directories(${LLVM_DIR}/lib)

set(CMAKE_CXX_COMPILER "${LLVM_DIR}/bin/clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 --cuda-gpu-arch=sm_30 -Wsign-compare -Wall -fno-rtti -pedantic -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS")

# CUDA
set(CUDA_DIR /usr/local/cuda)
include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib64)

# Instrumentation
add_subdirectory("instrument")

# Runtime
add_subdirectory("runtime")

set(SOURCE_FILES main.cpp kernel.cu general.h)
add_executable(cuda ${SOURCE_FILES})
target_link_libraries(cuda cudart dl rt pthread runtime)
