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
    emitter.store(
            emitter.getBuilder().CreatePointerCast(store->getPointerOperand(), this->context.getTypes().voidPtr()),
            this->context.getValues().int64(size), this->resolver.getAddressSpace(store),
            this->context.getValues().int64(this->context.getTypeMapper().mapItem(store->getValueOperand()->getType())),
            this->context.getValues().int32(debugIndex),
            this->castToInt64(store->getValueOperand(), emitter.getBuilder())
    );
}
void MemoryAccess::handleLoad(LoadInst* load, int32_t debugIndex)
{
    auto size = load->getModule()->getDataLayout().getTypeSizeInBits(
            load->getPointerOperand()->getType()->getPointerElementType()
    ) / 8;

    auto emitter = this->context.createEmitter(load->getNextNode());
    emitter.load(emitter.getBuilder().CreatePointerCast(load->getPointerOperand(), this->context.getTypes().voidPtr()),
                 this->context.getValues().int64(size),
                 this->resolver.getAddressSpace(load),
                 this->context.getValues().int64(this->context.getTypeMapper().mapItem(load->getType())),
                 this->context.getValues().int32(debugIndex),
                 this->castToInt64(load, emitter.getBuilder())
    );
}

Value* MemoryAccess::castToInt64(Value* value, IRBuilder<>& builder)
{
    auto type = value->getType();
    if (type->isFloatingPointTy())
    {
        return this->castToInt64(builder.CreateFPToUI(value, this->context.getTypes().int64()), builder);
    }
    if (type->isPointerTy())
    {
        return this->castToInt64(builder.CreateBitOrPointerCast(
                value, this->context.getTypes().int64()), builder);
    }
    if (type->isIntegerTy() && type->getIntegerBitWidth() != 64)
    {
        return this->castToInt64(builder.CreateIntCast(
                value, this->context.getTypes().int64(), false), builder);
    }

    return value;
}
