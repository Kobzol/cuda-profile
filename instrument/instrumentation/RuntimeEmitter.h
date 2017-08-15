#pragma once

#include <llvm/IR/IRBuilder.h>

namespace llvm {
    class Function;
    class Instruction;
    class Module;
    class Value;
}

class RuntimeEmitter
{
public:
    explicit RuntimeEmitter(llvm::Instruction* insertionPoint)
            : module(insertionPoint->getModule()), builder(insertionPoint)
    {

    }

    void store(llvm::Value* blockX,
               llvm::Value* blockY,
               llvm::Value* blockZ,
               llvm::Value* threadX,
               llvm::Value* threadY,
               llvm::Value* threadZ,
               llvm::Value* address,
               llvm::Value* size
    );
    void kernelStart();
    void kernelEnd(const std::string& kernelName);


    llvm::Value* readInt32(const std::string& name);

    llvm::IRBuilder<>& getBuilder()
    {
        return this->builder;
    }

private:
    llvm::Function* getStoreFunction();
    llvm::Function* getKernelStartFunction();
    llvm::Function* getKernelEndFunction();

    llvm::Module* module;
    llvm::IRBuilder<> builder;
};
