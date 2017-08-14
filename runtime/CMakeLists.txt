cmake_minimum_required(VERSION 3.7)

set(CMAKE_CXX_STANDARD 14)
set(SOURCE_FILES StoreRecord.h Runtime.h cudautil.h)

cuda_add_library(runtime STATIC ${SOURCE_FILES} OPTIONS "-arch sm_30;-std=c++11")
target_link_libraries(runtime ${CUDA_LIBRARIES})
