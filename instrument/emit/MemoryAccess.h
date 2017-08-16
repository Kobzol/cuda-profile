#pragma once

#include "../Context.h"

namespace llvm {
    class Function;
    class LoadInst;
    class Module;
    class StoreInst;
}


class StoreHandler
{
public:
    explicit StoreHandler(Context& context): context(context)
    {

    }

    void handleKernel(llvm::Function* kernel);

private:
    void handleStore(llvm::StoreInst* store);
    void handleLoad(llvm::LoadInst* load);

    Context& context;
};
