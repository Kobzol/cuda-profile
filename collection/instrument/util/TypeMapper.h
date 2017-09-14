#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>

namespace llvm {
    class Type;
}

class Context;

class TypeMapper
{
public:
    explicit TypeMapper(Context& context): context(context)
    {

    }

    uint64_t mapType(llvm::Type* type);
    std::vector<std::string> getTypes()
    {
        return this->types;
    }

private:
    Context& context;

    std::vector<std::string> types;
    std::unordered_map<std::string, uint64_t> typeMap;
};
