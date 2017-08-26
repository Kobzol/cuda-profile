#include "TypeMapper.h"
#include "../Context.h"

using namespace llvm;

uint64_t TypeMapper::mapType(Type* type)
{
    std::string name = this->context.getTypes().stringify(type);

    if (this->typeMap.find(name) == this->typeMap.end())
    {
        this->types.push_back(name);
        this->typeMap.insert({name, this->typeMap.size()});

        return this->typeMap.size() - 1;
    }

    return this->typeMap[name];
}
