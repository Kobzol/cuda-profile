#include "MemoryAlloc.h"

#include "RuntimeEmitter.h"
#include "../util/Types.h"
#include "../util/Values.h"

using namespace llvm;

Type* getMallocValueType(CallInst* malloc)
{
    // pass through two levels of indirection
    auto* bufferPtrType = dyn_cast<PointerType>(malloc->getOperand(0)->stripPointerCasts()->getType());
    auto* valuePtrType = dyn_cast<PointerType>(bufferPtrType->getElementType());

    return valuePtrType->getElementType();
}

void MemoryAlloc::handleCudaMalloc(CallInst* call)
{
    auto emitter = this->context.createEmitter(call->getNextNode());

    Value* addressLoad = emitter.getBuilder().CreateLoad(call->getOperand(0));

    Type* valueType = getMallocValueType(call);
    std::string typeStr = this->context.getTypes().print(valueType);
    GlobalVariable* typeCString = this->context.getValues().createGlobalCString(typeStr);

    emitter.malloc(addressLoad, call->getOperand(1),
                   this->context.getValues().int64(valueType->getPrimitiveSizeInBits() / 8), typeCString);
}

void MemoryAlloc::handleCudaFree(CallInst* call)
{
    auto emitter = this->context.createEmitter(call);
    emitter.free(call->getOperand(0));
}
