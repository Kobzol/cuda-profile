#include "RuntimeEmitter.h"
#include "../../runtime/prefix.h"

#include <llvm/IR/Module.h>
#include <iostream>

using namespace llvm;

static const std::string KERNEL_CONTEXT_TYPE = "KernelContext";


std::string RuntimeEmitter::runtimePrefix(const std::string& name)
{
    return PREFIX_STR + name;
}


void RuntimeEmitter::store(Value* address,
                           Value* size,
                           Value* addressSpace,
                           Value* type,
                           Value* debugIndex
)
{
    this->builder.CreateCall(this->getStoreFunction(), {
            address, size, addressSpace,
            type, debugIndex
    });
}
void RuntimeEmitter::load(Value* address,
                          Value* size,
                          Value* addressSpace,
                          Value* type,
                          Value* debugIndex
)
{
    this->builder.CreateCall(this->getLoadFunction(), {
            address, size, addressSpace,
            type, debugIndex
    });
}

void RuntimeEmitter::kernelStart(Value* kernelContext)
{
    this->builder.CreateCall(this->getKernelStartFunction(), {
            kernelContext
    });
}
void RuntimeEmitter::kernelEnd(Value* kernelContext)
{
    this->builder.CreateCall(this->getKernelEndFunction(), {
            kernelContext
    });
    this->builder.CreateCall(this->getDestroyKernelContextFunction(), {
            kernelContext
    });
}

void RuntimeEmitter::malloc(Value* address, Value* size,
                            Value* elementSize, Value* type)
{
    this->builder.CreateCall(this->getMallocFunction(), {
            address, size,
            elementSize, type
    });
}
void RuntimeEmitter::free(Value* address)
{
    this->builder.CreateCall(this->getFreeFunction(), {
            address
    });
}

Value* RuntimeEmitter::createKernelContext(Value* kernelName)
{
    auto type = this->context.getTypes().getCompositeType(KERNEL_CONTEXT_TYPE);
    auto alloc = this->builder.CreateAlloca(type);
    this->builder.CreateCall(this->getCreateKernelContextFunction(), {
            alloc,
            kernelName
    });

    return alloc;
}

Function* RuntimeEmitter::getStoreFunction()
{
    return cast<Function>(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("store"),
            this->context.getTypes().voidType(),
            this->context.getTypes().voidPtr(),
            this->context.getTypes().int64(),
            this->context.getTypes().int32(),
            this->context.getTypes().int8Ptr(),
            this->context.getTypes().int32(),
            nullptr));
}
llvm::Function* RuntimeEmitter::getLoadFunction()
{
    return cast<Function>(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("load"),
            this->context.getTypes().voidType(),
            this->context.getTypes().voidPtr(),
            this->context.getTypes().int64(),
            this->context.getTypes().int32(),
            this->context.getTypes().int8Ptr(),
            this->context.getTypes().int32(),
            nullptr));
}
Function* RuntimeEmitter::getKernelStartFunction()
{
    return cast<Function>(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("kernelStart"),
            this->context.getTypes().voidType(),
            this->context.getTypes().getCompositeType(KERNEL_CONTEXT_TYPE)->getPointerTo(),
            nullptr));
}
Function* RuntimeEmitter::getKernelEndFunction()
{
    return cast<Function>(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("kernelEnd"),
            this->context.getTypes().voidType(),
            this->context.getTypes().getCompositeType(KERNEL_CONTEXT_TYPE)->getPointerTo(),
            nullptr));
}
Function* RuntimeEmitter::getMallocFunction()
{
    return cast<Function>(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("malloc"),
            this->context.getTypes().voidType(),
            this->context.getTypes().int8Ptr(),
            this->context.getTypes().int64(),
            this->context.getTypes().int64(),
            this->context.getTypes().int8Ptr(),
            nullptr));
}
Function* RuntimeEmitter::getFreeFunction()
{
    return cast<Function>(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("free"),
            this->context.getTypes().voidType(),
            this->context.getTypes().int8Ptr(),
            nullptr));
}

Function* RuntimeEmitter::getCreateKernelContextFunction()
{
    return cast<Function>(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("initKernelContext"),
            this->context.getTypes().voidType(),
            this->context.getTypes().getCompositeType(KERNEL_CONTEXT_TYPE)->getPointerTo(),
            this->context.getTypes().int8Ptr(),
            nullptr));
}

Function* RuntimeEmitter::getDestroyKernelContextFunction()
{
    return cast<Function>(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("disposeKernelContext"),
            this->context.getTypes().voidType(),
            this->context.getTypes().getCompositeType(KERNEL_CONTEXT_TYPE)->getPointerTo(),
            nullptr));
}
