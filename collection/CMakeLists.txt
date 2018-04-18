cmake_minimum_required(VERSION 3.4)
project(cuda)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(CUDA REQUIRED)

find_package(LLVM CONFIG REQUIRED)
add_definitions(${LLVM_DEFINITIONS})

message("LLVM found at ${LLVM_DIR}")

find_package(ZLIB)
if (ZLIB_FOUND)
    add_definitions(-DCUPR_USE_ZLIB)
endif()

find_package(Protobuf)
if (PROTOBUF_FOUND)
    add_definitions(-DCUPR_USE_PROTOBUF)
    file(GLOB ProtoFiles "${CMAKE_CURRENT_SOURCE_DIR}/runtime/format/protobuf/*.proto")
    execute_process(COMMAND mkdir -p ${CMAKE_CURRENT_SOURCE_DIR}/tests/generated)
    execute_process(COMMAND
            ${Protobuf_PROTOC_EXECUTABLE}
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/runtime/format/protobuf
            --python_out=${CMAKE_CURRENT_SOURCE_DIR}/tests/generated
            ${ProtoFiles}
    )
endif()

find_package(CapnProto CONFIG)
if (CapnProto_FOUND)
    add_definitions(-DCUPR_USE_CAPNP)
endif()

include(ExternalProject)

set(RAPIDJSON_ROOT ${CMAKE_CURRENT_BINARY_DIR}/rapidjson)

# download rapidjson
ExternalProject_Add(rapidjson
        PREFIX ${RAPIDJSON_ROOT}
        GIT_REPOSITORY https://github.com/Tencent/rapidjson
        GIT_TAG master
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
)
include_directories(${RAPIDJSON_ROOT}/src/rapidjson/include)

set(CMAKE_CXX_COMPILER "${LLVM_TOOLS_BINARY_DIR}/clang++")
set(CUDA_HOST_COMPILER "g++")

# Instrumentation
add_subdirectory("instrument")

# Runtime
add_subdirectory("runtime")

# Memtracker
add_subdirectory("runtimetracker")

# Benchmark
add_subdirectory("benchmark")

# Samples
add_subdirectory("samples")

# Tests
add_custom_target(tests COMMAND py.test ${CMAKE_CURRENT_SOURCE_DIR}/tests)
add_dependencies(tests instrument)
