#pragma once

#include <cstdint>
#include "../Context.h"

namespace llvm {
    class Value;
}

class AddressSpaceResolver
{
public:
    explicit AddressSpaceResolver(Context& context);

    llvm::Value* getAddressSpace(llvm::Value* value);

private:
    llvm::Value* getAddressSpace(uint32_t space);

    Context& context;
};
