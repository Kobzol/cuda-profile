#include "MemoryAlloc.h"

#include "RuntimeEmitter.h"

using namespace llvm;


void MemoryAlloc::handleCudaMalloc(CallInst* call)
{
    RuntimeEmitter emitter(call->getNextNode());

    Value* addressLoad = emitter.getBuilder().CreateLoad(call->getOperand(0));
    emitter.malloc(addressLoad, call->getOperand(1));
}

void MemoryAlloc::handleCudaFree(CallInst* call)
{
    RuntimeEmitter emitter(call);
    emitter.free(call->getOperand(0));
}
