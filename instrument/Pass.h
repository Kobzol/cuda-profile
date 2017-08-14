#pragma once

#include <llvm/Pass.h>

struct CudaPass : public llvm::ModulePass
{
public:
    static char ID;

    CudaPass();

    virtual bool runOnModule(llvm::Module& module) override;

    void instrumentCuda(llvm::Module &module);
    void instrumentCpp(llvm::Module &module);
};
