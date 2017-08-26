#include "MemoryAccess.h"

#include <iostream>
#include <llvm/IR/Module.h>
#include <IR/ConstantsContext.h>

#include "RuntimeEmitter.h"
#include "../util/AddressSpaceResolver.h"
#include "../../runtime/AddressSpace.h"

using namespace llvm;


void MemoryAccess::handleStore(StoreInst* store, int32_t debugIndex)
{
    std::string type = this->context.getTypes().print(store->getValueOperand()->getType());
    auto* typeCString = this->context.getValues().createGlobalCString(type);

    auto emitter = this->context.createEmitter(store);
    emitter.store(emitter.getBuilder().CreatePointerCast(store->getPointerOperand(), this->context.getTypes().voidPtr()),
                  this->context.getValues().int64(store->getValueOperand()->getType()->getPrimitiveSizeInBits() / 8),
                  this->getAddressSpace(store->getPointerAddressSpace()),
                  typeCString,
                  this->context.getValues().int32(debugIndex)
    );
}
void MemoryAccess::handleLoad(LoadInst* load, int32_t debugIndex)
{
    std::string type = this->context.getTypes().print(load->getType());
    auto* typeCString = this->context.getValues().createGlobalCString(type);

    auto emitter = this->context.createEmitter(load);
    emitter.load(emitter.getBuilder().CreatePointerCast(load->getPointerOperand(), this->context.getTypes().voidPtr()),
                 this->context.getValues().int64(load->getPointerOperand()->getType()->getPrimitiveSizeInBits() / 8),
                 this->getAddressSpace(load->getPointerAddressSpace()),
                 typeCString,
                 this->context.getValues().int32(debugIndex)
    );
}
Value* MemoryAccess::getAddressSpace(uint32_t addressSpace)
{
    switch (addressSpace)
    {
        case 3: return this->context.getValues().int32(static_cast<uint32_t>(AddressSpace::Shared));
        case 4: return this->context.getValues().int32(static_cast<uint32_t>(AddressSpace::Constant));
        default: return this->context.getValues().int32(static_cast<uint32_t>(AddressSpace::Global));
    }
}
