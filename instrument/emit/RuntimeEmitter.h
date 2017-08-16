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
               llvm::Value* warpId,
               llvm::Value* address,
               llvm::Value* size,
               llvm::Value* type
    );
    void load(llvm::Value* blockX,
              llvm::Value* blockY,
              llvm::Value* blockZ,
              llvm::Value* threadX,
              llvm::Value* threadY,
              llvm::Value* threadZ,
              llvm::Value* warpId,
              llvm::Value* address,
              llvm::Value* size,
              llvm::Value* type
    );

    void kernelStart();
    void kernelEnd(llvm::Value* kernelName);

    void malloc(llvm::Value* address, llvm::Value* size, llvm::Value* type);
    void free(llvm::Value* address);

    llvm::Value* readInt32(const std::string& name);

    llvm::IRBuilder<>& getBuilder()
    {
        return this->builder;
    }

private:
    llvm::Function* getStoreFunction();
    llvm::Function* getLoadFunction();
    llvm::Function* getKernelStartFunction();
    llvm::Function* getKernelEndFunction();
    llvm::Function* getMallocFunction();
    llvm::Function* getFreeFunction();

    llvm::Module* module;
    llvm::IRBuilder<> builder;
};
