set(CUPR_SRC_DIR ${PROJECT_SOURCE_DIR})
set(CUPR_BUILD_DIR ${PROJECT_BINARY_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g --cuda-gpu-arch=sm_30 -Xclang -load \
-Xclang ${CUPR_BUILD_DIR}/instrument/libinstrument.so")

include_directories(${CUDA_INCLUDE_DIRS} ${CUPR_SRC_DIR}/include ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUPR_BUILD_DIR}/runtime)
link_libraries(cudart runtime)

add_subdirectory("bankConflict")
add_subdirectory("vectorAdd")

SET_TARGET_PROPERTIES(bankConflict PROPERTIES EXCLUDE_FROM_ALL True)
SET_TARGET_PROPERTIES(vectorAdd PROPERTIES EXCLUDE_FROM_ALL True)
