set(SOURCE_FILES
        CudaTimer.cpp
        Init.cpp
        Parameters.cpp
        RuntimeState.cpp
        format/CapnpTraceFormatter.cpp
        format/Emitter.cpp
        format/TraceFormatter.cpp
        format/JsonTraceFormatter.cpp
        format/ProtobufTraceFormatter.cpp
        format/warp/Warp.cpp
        format/warp/WarpGrouper.cpp
        tracking/Memtracker.cpp
)

# Compile Capnp definitions
if (CapnProto_FOUND)
    capnp_generate_cpp(capnpTraceSources capnpTraceHeaders format/capnp/cupr.capnp)
    message("Generated Capnp definitions")
endif()

# Compile Protobuf definitions
if (PROTOBUF_FOUND)
    PROTOBUF_GENERATE_CPP(protoTraceSources protoTraceHeaders ${ProtoFiles})
    message("Generated Protobuf definitions")
endif()

add_library(runtime SHARED ${SOURCE_FILES} ${capnpTraceSources} ${protoTraceSources})
target_link_libraries(runtime ${CUDA_LIBRARIES})
if (CapnProto_FOUND)
    target_link_libraries(runtime capnp kj)
endif()
if (PROTOBUF_FOUND)
    target_link_libraries(runtime ${PROTOBUF_LIBRARIES})
endif()
if (ZLIB_FOUND)
    target_link_libraries(runtime z)
endif()
target_include_directories(runtime PRIVATE ${CUDA_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
