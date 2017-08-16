#pragma once

#include <llvm/IR/IRBuilder.h>
#include "../Context.h"

namespace llvm {
    class Function;
    class Instruction;
    class Module;
    class Value;
}


class RuntimeEmitter
{
public:
    explicit RuntimeEmitter(Context& context, llvm::Instruction* insertionPoint)
            : context(context), builder(insertionPoint)
    {

    }

    void store(llvm::Value* address,
               llvm::Value* size,
               llvm::Value* type
    );
    void load(llvm::Value* address,
              llvm::Value* size,
              llvm::Value* type
    );

    void kernelStart();
    void kernelEnd(llvm::Value* kernelName);

    void malloc(llvm::Value* address, llvm::Value* size, llvm::Value* type);
    void free(llvm::Value* address);

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

    Context& context;
    llvm::IRBuilder<> builder;
};
