#include "RuntimeEmitter.h"
#include "../util/Types.h"
#include "../util/Values.h"
#include "../../runtime/prefix.h"

#include <llvm/IR/Module.h>


using namespace llvm;

std::string prefix(const std::string& name)
{
    return PREFIX_STR + name;
}

void RuntimeEmitter::store(Value* address,
                           Value* size,
                           Value* type
)
{
    this->builder.CreateCall(this->getStoreFunction(), {
            address, size, type
    });
}
void RuntimeEmitter::load(Value* address,
                          Value* size,
                          Value* type
)
{
    this->builder.CreateCall(this->getLoadFunction(), {
            address, size, type
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
            prefix("store"),
            this->context.getTypes().voidType(),
            this->context.getTypes().voidPtr(),
            this->context.getTypes().int64(),
            this->context.getTypes().int8Ptr(),
            nullptr));
}
llvm::Function* RuntimeEmitter::getLoadFunction()
{
    return cast<Function>(this->context.getModule()->getOrInsertFunction(
            prefix("load"),
            this->context.getTypes().voidType(),
            this->context.getTypes().voidPtr(),
            this->context.getTypes().int64(),
            this->context.getTypes().int8Ptr(),
            nullptr));
}
Function* RuntimeEmitter::getKernelStartFunction()
{
    return cast<Function>(this->context.getModule()->getOrInsertFunction(
            prefix("kernelStart"),
            this->context.getTypes().voidType(),
            nullptr));
}
Function* RuntimeEmitter::getKernelEndFunction()
{
    return cast<Function>(this->context.getModule()->getOrInsertFunction(
            prefix("kernelEnd"),
            this->context.getTypes().voidType(),
            this->context.getTypes().int8Ptr(),
            nullptr));
}
Function* RuntimeEmitter::getMallocFunction()
{
    return cast<Function>(this->context.getModule()->getOrInsertFunction(
            prefix("malloc"),
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
            prefix("free"),
            this->context.getTypes().voidType(),
            this->context.getTypes().int8Ptr(),
            nullptr));
}
