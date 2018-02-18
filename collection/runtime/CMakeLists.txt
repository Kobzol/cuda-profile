FILE(GLOB PROTOBUF_CXX_FILES format/protobuf/generated/*.cc)
set(SOURCE_FILES
        format/Emitter.cpp
        format/TraceFormatter.cpp
        format/JsonTraceFormatter.cpp
        format/ProtobufTraceFormatter.cpp
        tracking/Memtracker.cpp
        CudaTimer.cpp
        RuntimeState.cpp
        Init.cpp
        Parameters.cpp
)

add_library(runtime SHARED ${SOURCE_FILES} ${PROTOBUF_CXX_FILES})
target_link_libraries(runtime ${CUDA_LIBRARIES})
if (PROTOBUF_FOUND)
    target_link_libraries(runtime ${PROTOBUF_LIBRARIES})
endif(PROTOBUF_FOUND)
target_include_directories(runtime PRIVATE ${CUDA_INCLUDE_DIRS})
