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
    static std::string runtimePrefix(const std::string& name);

    explicit RuntimeEmitter(Context& context, llvm::Instruction* insertionPoint)
            : context(context), builder(insertionPoint)
    {

    }

    void store(llvm::Value* address, llvm::Value* size, llvm::Value* addressSpace, llvm::Value* type,
                   llvm::Value* debugIndex, llvm::Value* value);
    void load(llvm::Value* address,
              llvm::Value* size,
              llvm::Value* addressSpace,
              llvm::Value* type,
              llvm::Value* debugIndex,
              llvm::Value* value
    );

    void kernelStart(llvm::Value* kernelContext);
    void kernelEnd(llvm::Value* kernelContext);

    void malloc(llvm::Value* address, llvm::Value* size,
                llvm::Value* elementSize, llvm::Value* type,
                llvm::Value* name, llvm::Value* location);
    void free(llvm::Value* address);

    llvm::IRBuilder<>& getBuilder()
    {
        return this->builder;
    }

    llvm::Value* createKernelContext(llvm::Value* kernelName);

    void emitFirstThreadActions(const std::vector<llvm::GlobalVariable*>& sharedBuffers);

private:
    llvm::Function* getStoreFunction();
    llvm::Function* getLoadFunction();
    llvm::Function* getKernelStartFunction();
    llvm::Function* getKernelEndFunction();
    llvm::Function* getMallocFunction();
    llvm::Function* getFreeFunction();
    llvm::Function* getCreateKernelContextFunction();
    llvm::Function* getDestroyKernelContextFunction();
    llvm::Function* getIsFirstThreadFunction();
    llvm::Function* getMarkSharedBufferFunction();
    llvm::Function* getStoreDimensionsFunction();

    Context& context;
    llvm::IRBuilder<> builder;
};
