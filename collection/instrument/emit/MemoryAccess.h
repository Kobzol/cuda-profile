#pragma once

#include <llvm/IR/IRBuilder.h>
#include "../Context.h"
#include "../util/AddressSpaceResolver.h"

namespace llvm {
    class Function;
    class LoadInst;
    class StoreInst;
    class Value;
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

    llvm::Value* castToInt64(llvm::Value* value, llvm::IRBuilder<>& builder);
};
