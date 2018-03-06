#include "MemoryAccess.h"

#include <llvm/IR/Module.h>

#include "RuntimeEmitter.h"

using namespace llvm;

MemoryAccess::MemoryAccess(Context& context): context(context), resolver(context)
{

}

void MemoryAccess::handleStore(StoreInst* store, int32_t debugIndex)
{
    auto size = store->getModule()->getDataLayout().getTypeSizeInBits(
            store->getValueOperand()->getType()
    ) / 8;

    auto emitter = this->context.createEmitter(store);
    emitter.store(emitter.getBuilder().CreatePointerCast(store->getPointerOperand(), this->context.getTypes().voidPtr()),
                  this->context.getValues().int64(size),
                  this->resolver.getAddressSpace(store),
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
                 this->resolver.getAddressSpace(load),
                 this->context.getValues().int64(this->context.getTypeMapper().mapType(load->getType())),
                 this->context.getValues().int32(debugIndex)
    );
}
