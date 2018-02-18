#include "MemoryAccess.h"

#include <llvm/IR/Module.h>

#include "RuntimeEmitter.h"
#include "../util/AddressSpaceResolver.h"
#include "../../runtime/tracedata/AddressSpace.h"
#include "../util/LLVMAddressSpace.h"

using namespace llvm;


void MemoryAccess::handleStore(StoreInst* store, int32_t debugIndex)
{
    auto size = store->getModule()->getDataLayout().getTypeSizeInBits(
            store->getValueOperand()->getType()
    ) / 8;

    auto emitter = this->context.createEmitter(store);
    emitter.store(emitter.getBuilder().CreatePointerCast(store->getPointerOperand(), this->context.getTypes().voidPtr()),
                  this->context.getValues().int64(size),
                  this->getAddressSpace(store->getPointerAddressSpace()),
                  this->context.getValues().int64(this->context.getTypeMapper().mapType(store->getValueOperand()->getType())),
                  this->context.getValues().int32(debugIndex)
    );
}
void MemoryAccess::handleLoad(LoadInst* load, int32_t debugIndex)
{
    auto size = load->getModule()->getDataLayout().getTypeSizeInBits(
            load->getPointerOperand()->getType()->getPointerElementType()
    ) / 8;

    auto emitter = this->context.createEmitter(load);
    emitter.load(emitter.getBuilder().CreatePointerCast(load->getPointerOperand(), this->context.getTypes().voidPtr()),
                 this->context.getValues().int64(size),
                 this->getAddressSpace(load->getPointerAddressSpace()),
                 this->context.getValues().int64(this->context.getTypeMapper().mapType(load->getType())),
                 this->context.getValues().int32(debugIndex)
    );
}
Value* MemoryAccess::getAddressSpace(uint32_t addressSpace)
{
    switch (static_cast<LLVMAddressSpace>(addressSpace))
    {
        case LLVMAddressSpace::Shared:
            return this->context.getValues().int32(static_cast<uint32_t>(cupr::AddressSpace::Shared));
        case LLVMAddressSpace::Constant:
            return this->context.getValues().int32(static_cast<uint32_t>(cupr::AddressSpace::Constant));
        default:
            return this->context.getValues().int32(static_cast<uint32_t>(cupr::AddressSpace::Global));
    }
}
