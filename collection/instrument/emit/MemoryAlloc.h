#pragma once

#include <utility>

#include "../Context.h"

namespace llvm {
    class CallInst;
}


class MemoryAlloc
{
public:
    explicit MemoryAlloc(Context& context): context(context)
    {

    }

    void handleCudaMalloc(llvm::CallInst* call);
    void handleCudaFree(llvm::CallInst* call);

private:
    Context& context;
};
