cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 14)
set(SOURCE_FILES Runtime.h prefix.h format.h cudautil.h AccessRecord.h AllocRecord.h AddressSpace.h)

cuda_add_library(runtime STATIC ${SOURCE_FILES} OPTIONS "-arch=sm_30;-std=c++11")
target_link_libraries(runtime ${CUDA_LIBRARIES} ${PROTOBUF_LIBRARIES})
