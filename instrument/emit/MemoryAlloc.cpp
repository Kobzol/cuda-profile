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
    RuntimeEmitter emitter(call->getNextNode());

    Value* addressLoad = emitter.getBuilder().CreateLoad(call->getOperand(0));
    std::string type = Types::print(getMallocValueType(call));
    GlobalVariable* typeCString = Values::createGlobalCString(
            call->getModule(), "__cuProfileType_" + type, type);

    emitter.malloc(addressLoad, call->getOperand(1), typeCString);
}

void MemoryAlloc::handleCudaFree(CallInst* call)
{
    RuntimeEmitter emitter(call);
    emitter.free(call->getOperand(0));
}
