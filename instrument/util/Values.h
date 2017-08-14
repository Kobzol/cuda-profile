#pragma once

#include <cstdint>

namespace llvm {
    class Value;
    class Module;
}

class Values
{
public:
    static llvm::Value* int32(llvm::Module* module, int32_t value);
    static llvm::Value* int64(llvm::Module* module, int64_t value);
};
