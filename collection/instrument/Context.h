#pragma once

#include "util/Types.h"
#include "util/Values.h"
#include "util/Mapper.h"

namespace llvm {
    class Instruction;
    class Module;
}

class RuntimeEmitter;

class Context
{
public:
    explicit Context(): values(this->types), typeMapper([this](const llvm::Type* type) {
        return this->getTypes().stringify(type);
    }), nameMapper([this](const std::string& type) {
        return type;
    })
    {

    }

    void setModule(llvm::Module* module);

    Types& getTypes()
    {
        return this->types;
    }
    Values& getValues()
    {
        return this->values;
    }
    Mapper<llvm::Type*>& getTypeMapper()
    {
        return this->typeMapper;
    }
    Mapper<std::string>& getNameMapper()
    {
        return this->nameMapper;
    }
    llvm::Module* getModule()
    {
        return this->module;
    }

    RuntimeEmitter createEmitter(llvm::Instruction* insertionPoint);

private:
    llvm::Module* module = nullptr;
    Types types;
    Values values;
    Mapper<llvm::Type*> typeMapper;
    Mapper<std::string> nameMapper;
};
