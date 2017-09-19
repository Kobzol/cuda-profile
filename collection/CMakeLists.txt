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

# Device runtime
add_subdirectory("device")

# Tests
add_custom_target(tests COMMAND py.test ${CMAKE_CURRENT_SOURCE_DIR}/tests)
add_dependencies(tests instrument)

# Compile protobuf definitions
FILE(GLOB PROTOBUF_PROTO_FILES runtime/protobuf/*.proto)
execute_process(COMMAND
        ${Protobuf_PROTOC_EXECUTABLE}
        --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/runtime/protobuf
        --cpp_out=${CMAKE_CURRENT_SOURCE_DIR}/runtime/protobuf/generated
        ${PROTOBUF_PROTO_FILES}
)
execute_process(COMMAND
        ${Protobuf_PROTOC_EXECUTABLE}
        --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/runtime/protobuf
        --python_out=${CMAKE_CURRENT_SOURCE_DIR}/tests/generated
        ${PROTOBUF_PROTO_FILES}
)

# CUDA program
#FILE(GLOB PROTOBUF_SRC runtime/protobuf/generated/*.pb.cc)
#set(SOURCE_FILES main.cpp kernel.cu general.h ${PROTOBUF_SRC})
#link_directories("/usr/local/cuda/lib64")
#add_executable(cuda ${SOURCE_FILES})
#target_link_libraries(cuda runtime cudart rt dl ${PROTOBUF_LIBRARIES})
