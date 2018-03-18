set(SOURCE_FILES
        Context.cpp
        Parameters.cpp
        Pass.cpp
        emit/MemoryAccess.cpp
        emit/MemoryAlloc.cpp
        emit/Kernel.cpp
        emit/KernelInit.cpp
        emit/KernelLaunch.cpp
        emit/RuntimeEmitter.cpp
        util/AddressSpaceResolver.cpp
        util/DebugExtractor.cpp
        util/DebugInfo.cpp
        util/Demangler.cpp
        util/FunctionContentLoader.cpp
        util/FunctionUtils.cpp
        util/RegexFilter.cpp
        util/StringUtils.cpp
        util/Types.cpp
        util/Values.cpp
)

add_library(instrument MODULE RegisterPass.cpp ${SOURCE_FILES})
target_include_directories(instrument PRIVATE ${LLVM_INCLUDE_DIRS})
set_target_properties(instrument PROPERTIES COMPILE_FLAGS "-fno-rtti")
