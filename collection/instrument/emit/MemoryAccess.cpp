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
                  this->getAddressSpace(store),
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
                 this->getAddressSpace(load),
                 this->context.getValues().int64(this->context.getTypeMapper().mapType(load->getType())),
                 this->context.getValues().int32(debugIndex)
    );
}
Value* MemoryAccess::getAddressSpace(Value* value)
{
    if (auto spaceCast = dyn_cast<AddrSpaceCastInst>(value))
    {
        return this->getAddressSpace(spaceCast->getSrcAddressSpace());
    }
    else if (auto constExpr = dyn_cast<ConstantExpr>(value))
    {
        return this->getAddressSpace(constExpr->getAsInstruction());
    }
    else if (auto gep = dyn_cast<GetElementPtrInst>(value))
    {
        return this->getAddressSpace(gep->getOperand(0));
    }
    else if (auto load = dyn_cast<LoadInst>(value))
    {
        if (!isa<AllocaInst>(load->getPointerOperand()) && load->getPointerAddressSpace() == 0)
        {
            return this->getAddressSpace(load->getPointerOperand());
        }

        return this->getAddressSpace(load->getPointerAddressSpace());
    }
    else if (auto store = dyn_cast<StoreInst>(value))
    {
        if (!isa<AllocaInst>(store->getPointerOperand()) && store->getPointerAddressSpace() == 0)
        {
            return this->getAddressSpace(store->getPointerOperand());
        }

        return this->getAddressSpace(store->getPointerAddressSpace());
    }

    return this->context.getValues().int32(static_cast<uint32_t>(cupr::AddressSpace::Global));
}
Value* MemoryAccess::getAddressSpace(uint32_t space)
{
    switch (static_cast<LLVMAddressSpace>(space))
    {
        case LLVMAddressSpace::Shared:
            return this->context.getValues().int32(static_cast<uint32_t>(cupr::AddressSpace::Shared));
        case LLVMAddressSpace::Constant:
            return this->context.getValues().int32(static_cast<uint32_t>(cupr::AddressSpace::Constant));
        default:
            return this->context.getValues().int32(static_cast<uint32_t>(cupr::AddressSpace::Global));
    }
}
