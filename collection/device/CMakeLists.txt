cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 14)
set(SOURCE_FILES Runtime.h)

add_library(deviceruntime STATIC ${SOURCE_FILES})
target_link_libraries(deviceruntime ${CUDA_LIBRARIES} ${PROTOBUF_LIBRARIES})
target_include_directories(deviceruntime PRIVATE /home/kobzol/projects/diplomka/collection/runtime /usr/local/cuda/include)
set_target_properties(deviceruntime PROPERTIES LANGUAGE CXX)
set_target_properties(deviceruntime PROPERTIES LINKER_LANGUAGE CXX)
