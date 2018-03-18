#pragma once

#include "../Context.h"

namespace llvm {
    class Function;
    class GlobalVariable;
}


class KernelInit
{
public:
    explicit KernelInit(Context& context): context(context)
    {

    }

    void handleKernelInit(llvm::Function* function, const std::vector<llvm::GlobalVariable*>& sharedBuffers);

private:
    Context& context;
};
