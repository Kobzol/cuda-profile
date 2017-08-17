cmake_minimum_required(VERSION 3.7)

set(CMAKE_CXX_STANDARD 14)

include_directories("/home/kobzol/libraries/llvm-4.0.1.src/lib")

set(SOURCE_FILES Pass.cpp
    emit/MemoryAccess.cpp emit/KernelLaunch.cpp emit/RuntimeEmitter.cpp emit/MemoryAlloc.cpp
    util/Types.cpp util/Values.cpp util/Demangler.cpp util/StringUtils.cpp util/StringUtils.h Context.cpp Context.h emit/Kernel.cpp emit/Kernel.h)

add_library(llvmCuda MODULE RegisterPass.cpp ${SOURCE_FILES})
target_include_directories(llvmCuda PRIVATE ${LLVM_INCLUDE_DIRS})
set_target_properties(llvmCuda PROPERTIES COMPILE_FLAGS "-fno-rtti")
