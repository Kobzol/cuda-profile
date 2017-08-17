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

    void handleStore(llvm::StoreInst* store, size_t debugIndex);
    void handleLoad(llvm::LoadInst* load, size_t debugIndex);

private:
    Context& context;
};
