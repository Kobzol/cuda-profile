#pragma once

#include "../Context.h"

namespace llvm {
    class Function;
    class LoadInst;
    class StoreInst;
}


class MemoryAccess
{
public:
    explicit MemoryAccess(Context& context): context(context)
    {

    }

    void handleStore(llvm::StoreInst* store);
    void handleLoad(llvm::LoadInst* load);

private:
    Context& context;
};
