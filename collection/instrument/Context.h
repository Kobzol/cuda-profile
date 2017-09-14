#pragma once

#include "util/Types.h"
#include "util/Values.h"
#include "util/TypeMapper.h"

namespace llvm {
    class Instruction;
    class Module;
}

class RuntimeEmitter;

class Context
{
public:
    Context(): values(this->types), typeMapper(*this)
    {

    }

    explicit Context(llvm::Module* module): values(this->types), typeMapper(*this)
    {
        this->setModule(module);
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
    TypeMapper& getTypeMapper()
    {
        return this->typeMapper;
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
    TypeMapper typeMapper;
};
