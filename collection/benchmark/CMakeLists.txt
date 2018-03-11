set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --cuda-gpu-arch=sm_30 -Xclang -load -Xclang ${CMAKE_BINARY_DIR}/instrument/libinstrument.so")

file(GLOB CUDA_SOURCES samples/*.cu)
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CXX)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories("${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc")
link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")

set(SOURCE_FILES
        Benchmark.cpp
        ${CUDA_SOURCES}
)

add_executable(benchmark main.cpp ${SOURCE_FILES})
target_link_libraries(benchmark cudart runtime)

add_executable(benchmark-profiled main.cpp ${SOURCE_FILES})
add_dependencies(benchmark-profiled instrument runtime)
target_compile_definitions(benchmark-profiled PRIVATE PROFILE)
target_link_libraries(benchmark-profiled cudart runtime)
