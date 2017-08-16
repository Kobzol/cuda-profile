#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#include "Types.h"

namespace llvm {
    class GlobalVariable;
    class Module;
    class Value;
}


class Values
{
public:
    explicit Values(Types& types): types(types)
    {

    }

    void setModule(llvm::Module* module)
    {
        this->module = module;
        this->cStringMap.clear();
    }

    llvm::Value* int32(int32_t value);
    llvm::Value* int64(int64_t value);

    llvm::GlobalVariable* createGlobalCString(const std::string& value);

private:
    std::string generateGlobalName();

    Types& types;
    llvm::Module* module = nullptr;

    std::unordered_map<std::string, llvm::GlobalVariable*> cStringMap;
    size_t globalCounter = 0;
};
