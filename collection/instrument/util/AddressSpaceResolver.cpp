#include "AddressSpaceResolver.h"

#include "../../runtime/tracedata/AddressSpace.h"
#include "LLVMAddressSpace.h"

#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>

using namespace llvm;

AddressSpaceResolver::AddressSpaceResolver(Context& context) : context(context)
{

}

Value* AddressSpaceResolver::getAddressSpace(Value* value)
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

Value* AddressSpaceResolver::getAddressSpace(uint32_t space)
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
