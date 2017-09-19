cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 14)

FILE(GLOB PROTOBUF_CXX_FILES ./protobuf/generated/*.cc)
set(SOURCE_FILES Emitter.cpp CudaTimer.cpp Formatter.cpp Parameters.cpp RuntimeState.cpp AccessRecord.h AllocRecord.h AddressSpace.h)

add_library(runtime SHARED ${SOURCE_FILES} ${PROTOBUF_CXX_FILES})
target_link_libraries(runtime ${CUDA_LIBRARIES} ${PROTOBUF_LIBRARIES})
target_include_directories(runtime PRIVATE ${CUDA_INCLUDE_DIRS})
target_compile_definitions(runtime PRIVATE CUPR_USE_PROTOBUF=1)
