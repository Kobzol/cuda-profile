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
    std::string type = this->context.getTypes().print(getMallocValueType(call));
    GlobalVariable* typeCString = this->context.getValues().createGlobalCString(type);

    emitter.malloc(addressLoad, call->getOperand(1), typeCString);
}

void MemoryAlloc::handleCudaFree(CallInst* call)
{
    auto emitter = this->context.createEmitter(call);
    emitter.free(call->getOperand(0));
}
