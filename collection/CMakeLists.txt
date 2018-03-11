cmake_minimum_required(VERSION 3.4)
project(cuda)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(CUDA REQUIRED)

#set(LLVM_DIR "/home/kobzol/libraries/llvm-5.0.1-build/lib/cmake/llvm")
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

# Compile protobuf definitions
if (PROTOBUF_FOUND)
    FILE(GLOB PROTOBUF_PROTO_FILES runtime/format/protobuf/*.proto)

    execute_process(COMMAND mkdir -p ${CMAKE_CURRENT_SOURCE_DIR}/runtime/format/protobuf/generated)
    execute_process(COMMAND
            protoc
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/runtime/format/protobuf
            --cpp_out=${CMAKE_CURRENT_SOURCE_DIR}/runtime/format/protobuf/generated
            ${PROTOBUF_PROTO_FILES}
    )
    execute_process(COMMAND mkdir -p ${CMAKE_CURRENT_SOURCE_DIR}/tests/generated)
    execute_process(COMMAND
            protoc
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/runtime/format/protobuf
            --python_out=${CMAKE_CURRENT_SOURCE_DIR}/tests/generated
            ${PROTOBUF_PROTO_FILES}
    )
    message("Generated Protobuf definitions")
endif (PROTOBUF_FOUND)

# Instrumentation
add_subdirectory("instrument")

# Runtime
add_subdirectory("runtime")

# Memtracker
add_subdirectory("runtimetracker")

# Benchmark
add_subdirectory("benchmark")

# Tests
add_custom_target(tests COMMAND py.test ${CMAKE_CURRENT_SOURCE_DIR}/tests)
add_dependencies(tests instrument)
