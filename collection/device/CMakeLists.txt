cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 14)
set(SOURCE_FILES phantom.cu)

include_directories("/home/kobzol/projects/diplomka/collection/runtime")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set_source_files_properties(Runtime.h PROPERTIES LANGUAGE CXX)
set_source_files_properties(Runtime.h PROPERTIES LINKER_LANGUAGE CXX)

cuda_add_library(deviceruntime ${SOURCE_FILES} OPTIONS "-I/home/kobzol/projects/diplomka/collection/runtime")
