cmake_minimum_required(VERSION 3.4)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(CUPR_SRC_DIR "../../"
        REALPATH BASE_DIR "${PROJECT_SOURCE_DIR}")

set(CUPR_BUILD_DIR "${CUPR_SRC_DIR}/cmake-build-debug")

find_package(CUDA REQUIRED)

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g --cuda-gpu-arch=sm_30 -Xclang -load \
-Xclang ${CUPR_BUILD_DIR}/instrument/libinstrument.so")

set_source_files_properties(source.cu PROPERTIES LANGUAGE CXX)

include_directories(${CUDA_INCLUDE_DIRS})
link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
link_directories("${CUPR_BUILD_DIR}/runtime")

add_executable(bankConflict source.cu)
target_link_libraries(bankConflict cudart runtime)
target_include_directories(bankConflict PRIVATE ${CUPR_SRC_DIR}/include)
