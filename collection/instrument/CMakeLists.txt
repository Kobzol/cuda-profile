cmake_minimum_required(VERSION 3.8)

set(SOURCE_FILES
        Context.cpp
        Pass.cpp
        emit/MemoryAccess.cpp
        emit/KernelLaunch.cpp
        emit/RuntimeEmitter.cpp
        emit/MemoryAlloc.cpp
        emit/Kernel.cpp
        util/Types.cpp
        util/Values.cpp
        util/Demangler.cpp
        util/StringUtils.cpp
        util/DebugInfo.cpp
        util/DebugExtractor.cpp
        util/FunctionUtils.cpp
        util/TypeMapper.cpp
)

add_library(instrument MODULE RegisterPass.cpp ${SOURCE_FILES})
target_include_directories(instrument PRIVATE ${LLVM_INCLUDE_DIRS} "/home/kobzol/libraries/llvm-4.0.1.src/lib")
set_target_properties(instrument PROPERTIES COMPILE_FLAGS "-fno-rtti")
