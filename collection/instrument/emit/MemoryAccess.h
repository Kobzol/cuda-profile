#pragma once

#include "../Context.h"
#include "../util/AddressSpaceResolver.h"

namespace llvm {
    class Function;
    class LoadInst;
    class StoreInst;
}


class MemoryAccess
{
public:
    explicit MemoryAccess(Context& context);

    void handleStore(llvm::StoreInst* store, int32_t debugIndex);
    void handleLoad(llvm::LoadInst* load, int32_t debugIndex);

private:
    AddressSpaceResolver resolver;
    Context& context;
};
