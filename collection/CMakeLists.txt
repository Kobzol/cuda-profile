cmake_minimum_required(VERSION 3.8)
project(cuda)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(CUDA REQUIRED)

#set(LLVM_DIR "/home/kobzol/libraries/llvm-5.0.0-build/lib/cmake/llvm")
find_package(LLVM CONFIG REQUIRED)
add_definitions(${LLVM_DEFINITIONS})

message("LLVM found at ${LLVM_DIR}")

find_package(Protobuf)
if (PROTOBUF_FOUND)
    add_definitions(-DCUPR_USE_PROTOBUF)
endif (PROTOBUF_FOUND)

find_package(ZLIB)
if (ZLIB_FOUND)
    add_definitions(-DCUPR_USE_ZLIB)
endif (ZLIB_FOUND)

set(CMAKE_CXX_COMPILER "${LLVM_TOOLS_BINARY_DIR}/clang++")
set(CUDA_HOST_COMPILER "g++")

# Instrumentation
add_subdirectory("instrument")

# Runtime
add_subdirectory("runtime")

# Device runtime
add_subdirectory("device")

# Memtracker
add_subdirectory("runtimetracker")

# Tests
add_custom_target(tests COMMAND py.test ${CMAKE_CURRENT_SOURCE_DIR}/tests)
add_dependencies(tests instrument)

# Compile protobuf definitions
FILE(GLOB PROTOBUF_PROTO_FILES runtime/format/protobuf/*.proto)
execute_process(COMMAND
        ${Protobuf_PROTOC_EXECUTABLE}
        --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/runtime/format/protobuf
        --cpp_out=${CMAKE_CURRENT_SOURCE_DIR}/runtime/format/protobuf/generated
        ${PROTOBUF_PROTO_FILES}
)
execute_process(COMMAND
        ${Protobuf_PROTOC_EXECUTABLE}
        --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/runtime/format/protobuf
        --python_out=${CMAKE_CURRENT_SOURCE_DIR}/tests/generated
        ${PROTOBUF_PROTO_FILES}
)
