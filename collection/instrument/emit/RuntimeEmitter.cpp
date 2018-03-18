#include "RuntimeEmitter.h"
#include "../../runtime/Prefix.h"

#include <llvm/IR/Module.h>

using namespace llvm;

static const std::string KERNEL_CONTEXT_TYPE = "cupr::KernelContext";

static Function* toFunction(Constant* constant)
{
    return cast<Function>(constant->stripPointerCasts());
}


std::string RuntimeEmitter::runtimePrefix(const std::string& name)
{
    return CU_PREFIX_STR + name;
}


void RuntimeEmitter::store(Value* address, Value* size, Value* addressSpace,
                           Value* type, Value* debugIndex, Value* value)
{
    this->builder.CreateCall(this->getStoreFunction(), {
            address, size, addressSpace,
            type, debugIndex, value
    });
}
void RuntimeEmitter::load(Value* address,
                          Value* size,
                          Value* addressSpace,
                          Value* type,
                          Value* debugIndex,
                          Value* value
)
{
    this->builder.CreateCall(this->getLoadFunction(), {
            address, size, addressSpace,
            type, debugIndex, value
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
                            Value* elementSize, Value* type,
                            Value* name, Value* location)
{
    this->builder.CreateCall(this->getMallocFunction(), {
            address, size,
            elementSize, type,
            name, location
    });
}
void RuntimeEmitter::free(Value* address)
{
    this->builder.CreateCall(this->getFreeFunction(), {
            address
    });
}

void RuntimeEmitter::markSharedBuffer(Value* address, Value* size, Value* elementSize, Value* type, Value* name)
{
    this->builder.CreateCall(this->getMarkSharedBufferFunction(), {
            address, size, elementSize,
            type, name
    });
}
void RuntimeEmitter::storeKernelDimensions()
{
    this->builder.CreateCall(this->getStoreDimensionsFunction());
}
Instruction* RuntimeEmitter::isFirstThread()
{
    return this->builder.CreateCall(this->getIsFirstThreadFunction());
}

Instruction* RuntimeEmitter::barrier()
{
    return this->builder.CreateCall(this->context.getModule()->getOrInsertFunction(
            "llvm.nvvm.barrier0",
            this->context.getTypes().voidType(),
            nullptr)
    );
}

void RuntimeEmitter::setInsertPoint(BasicBlock* block)
{
    this->builder.SetInsertPoint(block);
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
    return toFunction(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("store"),
            this->context.getTypes().voidType(),
            this->context.getTypes().voidPtr(),
            this->context.getTypes().int64(),
            this->context.getTypes().int32(),
            this->context.getTypes().int64(),
            this->context.getTypes().int32(),
            this->context.getTypes().int64(),
            nullptr));
}
Function* RuntimeEmitter::getLoadFunction()
{
    return toFunction(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("load"),
            this->context.getTypes().voidType(),
            this->context.getTypes().voidPtr(),
            this->context.getTypes().int64(),
            this->context.getTypes().int32(),
            this->context.getTypes().int64(),
            this->context.getTypes().int32(),
            this->context.getTypes().int64(),
            nullptr));
}
Function* RuntimeEmitter::getKernelStartFunction()
{
    return toFunction(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("kernelStart"),
            this->context.getTypes().voidType(),
            this->context.getTypes().getCompositeType(KERNEL_CONTEXT_TYPE)->getPointerTo(),
            nullptr));
}
Function* RuntimeEmitter::getKernelEndFunction()
{
    return toFunction(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("kernelEnd"),
            this->context.getTypes().voidType(),
            this->context.getTypes().getCompositeType(KERNEL_CONTEXT_TYPE)->getPointerTo(),
            nullptr));
}
Function* RuntimeEmitter::getMallocFunction()
{
    return toFunction(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("malloc"),
            this->context.getTypes().voidType(),
            this->context.getTypes().int8Ptr(),
            this->context.getTypes().int64(),
            this->context.getTypes().int64(),
            this->context.getTypes().int8Ptr(),
            this->context.getTypes().int8Ptr(),
            this->context.getTypes().int8Ptr(),
            nullptr));
}
Function* RuntimeEmitter::getFreeFunction()
{
    return toFunction(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("free"),
            this->context.getTypes().voidType(),
            this->context.getTypes().int8Ptr(),
            nullptr));
}

Function* RuntimeEmitter::getCreateKernelContextFunction()
{
    return toFunction(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("initKernelContext"),
            this->context.getTypes().voidType(),
            this->context.getTypes().getCompositeType(KERNEL_CONTEXT_TYPE)->getPointerTo(),
            this->context.getTypes().int8Ptr(),
            nullptr));
}
Function* RuntimeEmitter::getDestroyKernelContextFunction()
{
    return toFunction(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("disposeKernelContext"),
            this->context.getTypes().voidType(),
            this->context.getTypes().getCompositeType(KERNEL_CONTEXT_TYPE)->getPointerTo(),
            nullptr));
}

Function* RuntimeEmitter::getIsFirstThreadFunction()
{
    return toFunction(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("isFirstThread"),
            this->context.getTypes().boolType(),
            nullptr));
}
Function* RuntimeEmitter::getMarkSharedBufferFunction()
{
    return toFunction(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("markSharedBuffer"),
            this->context.getTypes().voidType(),
            this->context.getTypes().int8Ptr(),
            this->context.getTypes().int64(),
            this->context.getTypes().int64(),
            this->context.getTypes().int64(),
            this->context.getTypes().int64(),
            nullptr));
}
Function* RuntimeEmitter::getStoreDimensionsFunction()
{
    return toFunction(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("storeDimensions"),
            this->context.getTypes().voidType(),
            nullptr));
}
