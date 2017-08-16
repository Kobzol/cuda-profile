#pragma once

#include "../Context.h"

namespace llvm {
    class CallInst;
}


class KernelLaunch
{
public:
    explicit KernelLaunch(Context& context): context(context)
    {

    }

    void handleKernelLaunch(llvm::CallInst* callSite);

private:
    Context& context;
};
