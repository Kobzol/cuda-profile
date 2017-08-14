cmake_minimum_required(VERSION 3.7)

set(CMAKE_CXX_STANDARD 14)

include_directories("/home/kobzol/libraries/llvm-4.0.1.src/lib")

set(SOURCE_FILES Pass.cpp Store.h Store.cpp util/Types.cpp util/Types.h util/Values.cpp util/Values.h)

add_library(llvmCuda MODULE RegisterPass.cpp ${SOURCE_FILES})
target_include_directories(llvmCuda PRIVATE ${LLVM_INCLUDE_DIRS})
set_target_properties(llvmCuda PROPERTIES COMPILE_FLAGS "-fno-rtti")
