set(SOURCE_FILES RuntimeTracker.cpp)

add_library(runtimetracker SHARED ${SOURCE_FILES})
target_link_libraries(runtimetracker ${CUDA_LIBRARIES})
target_include_directories(runtimetracker PRIVATE ${CUDA_INCLUDE_DIRS})
