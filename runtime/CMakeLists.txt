cmake_minimum_required(VERSION 3.7)

set(CMAKE_CXX_STANDARD 14)
set(SOURCE_FILES Runtime.h prefix.h cudautil.h StoreRecord.h)

cuda_add_library(runtime STATIC ${SOURCE_FILES} OPTIONS "-arch=sm_30;-std=c++11")
target_link_libraries(runtime ${CUDA_LIBRARIES})
