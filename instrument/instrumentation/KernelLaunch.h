#pragma once

#include <llvm/IR/Instructions.h>

namespace llvm {
    class CallInst;
}

class KernelLaunch
{
public:
    void handleKernelLaunch(llvm::CallInst* callSite);
};
