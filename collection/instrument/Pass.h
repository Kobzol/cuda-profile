#pragma once

#include <llvm/Pass.h>
#include <unordered_map>
#include "Context.h"

namespace llvm {
    class CallInst;
    class Function;
    class Module;
}

struct CudaPass : public llvm::ModulePass
{
public:
    static char ID;

    CudaPass();

    bool runOnModule(llvm::Module& module) override;

private:
    void instrumentCuda(llvm::Module& module);
    void instrumentCpp(llvm::Module& module);

    llvm::Function* augmentKernel(llvm::Function* fn);

    std::unordered_map<llvm::Function*, llvm::Function*> kernelMap;

    void handleFunctionCall(llvm::CallInst* call);

    Context context;

    bool isInstrumentableCuda(llvm::Module& module);

    bool isInstrumentableCpp(llvm::Module& module);
};
