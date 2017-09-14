#include "Context.h"

#include "emit/RuntimeEmitter.h"

using namespace llvm;


void Context::setModule(llvm::Module* module)
{
    this->module = module;
    this->types.setModule(module);
    this->values.setModule(module);
}

RuntimeEmitter Context::createEmitter(Instruction* insertionPoint)
{
    return RuntimeEmitter(*this, insertionPoint);
}
