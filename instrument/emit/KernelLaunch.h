#pragma once

namespace llvm {
    class CallInst;
}

class KernelLaunch
{
public:
    void handleKernelLaunch(llvm::CallInst* callSite);
};
