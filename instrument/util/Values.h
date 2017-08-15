#pragma once

#include <cstdint>
#include <string>

namespace llvm {
    class GlobalVariable;
    class Module;
    class Value;
}

class Values
{
public:
    static llvm::Value* int32(llvm::Module* module, int32_t value);
    static llvm::Value* int64(llvm::Module* module, int64_t value);

    static llvm::GlobalVariable* createGlobalCString(llvm::Module* module,
                                                     const std::string& name, const std::string& value);
};
