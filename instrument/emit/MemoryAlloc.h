#pragma once

namespace llvm {
    class CallInst;
}

class MemoryAlloc
{
public:
    void handleCudaMalloc(llvm::CallInst* call);
    void handleCudaFree(llvm::CallInst* call);
};
