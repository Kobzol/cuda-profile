#pragma once

#include "util/Types.h"
#include "util/Values.h"

namespace llvm {
    class Instruction;
    class Module;
}

class RuntimeEmitter;

class Context
{
public:
    Context(): values(this->types)
    {

    }

    explicit Context(llvm::Module* module): values(this->types)
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
    llvm::Module* getModule()
    {
        return this->module;
    }

    RuntimeEmitter createEmitter(llvm::Instruction* insertionPoint);

private:
    llvm::Module* module = nullptr;
    Types types;
    Values values;
};
