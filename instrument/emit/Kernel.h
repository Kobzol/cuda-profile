#pragma once

#include "../Context.h"

namespace llvm {
    class Function;
    class LoadInst;
    class StoreInst;
}


class Kernel
{
public:
    explicit Kernel(Context& context): context(context)
    {

    }

    void handleKernel(llvm::Function* function);

private:
    Context& context;

    void handleStore(llvm::StoreInst* store);
    void handleLoad(llvm::LoadInst* load);

    void traverseInstructions(llvm::Function* function);
};
