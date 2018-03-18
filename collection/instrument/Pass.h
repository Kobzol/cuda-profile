#pragma once

#include <llvm/Pass.h>
#include <unordered_map>
#include "Context.h"
#include "util/RegexFilter.h"

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

    void handleFunctionCall(llvm::CallInst* call);
    bool isInstrumentableCuda(llvm::Module& module);
    bool isInstrumentableCpp(llvm::Module& module);

    Context context;
    std::unordered_map<llvm::Function*, llvm::Function*> kernelMap;
    RegexFilter filter;
};
