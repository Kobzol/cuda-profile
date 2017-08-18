#include "RuntimeEmitter.h"
#include "../../runtime/prefix.h"

#include <llvm/IR/Module.h>


using namespace llvm;

std::string RuntimeEmitter::runtimePrefix(const std::string& name)
{
    return PREFIX_STR + name;
}

void RuntimeEmitter::store(Value* address,
                           Value* size,
                           Value* type,
                           Value* debugIndex
)
{
    this->builder.CreateCall(this->getStoreFunction(), {
            address, size, type, debugIndex
    });
}
void RuntimeEmitter::load(Value* address,
                          Value* size,
                          Value* type,
                          Value* debugIndex
)
{
    this->builder.CreateCall(this->getLoadFunction(), {
            address, size, type, debugIndex
    });
}

void RuntimeEmitter::kernelStart()
{
    this->builder.CreateCall(this->getKernelStartFunction());
}
void RuntimeEmitter::kernelEnd(Value* kernelName)
{
    this->builder.CreateCall(this->getKernelEndFunction(), {
        kernelName
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

Function* RuntimeEmitter::getStoreFunction()
{
    return cast<Function>(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("store"),
            this->context.getTypes().voidType(),
            this->context.getTypes().voidPtr(),
            this->context.getTypes().int64(),
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
            this->context.getTypes().int8Ptr(),
            this->context.getTypes().int32(),
            nullptr));
}
Function* RuntimeEmitter::getKernelStartFunction()
{
    return cast<Function>(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("kernelStart"),
            this->context.getTypes().voidType(),
            nullptr));
}
Function* RuntimeEmitter::getKernelEndFunction()
{
    return cast<Function>(this->context.getModule()->getOrInsertFunction(
            RuntimeEmitter::runtimePrefix("kernelEnd"),
            this->context.getTypes().voidType(),
            this->context.getTypes().int8Ptr(),
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
