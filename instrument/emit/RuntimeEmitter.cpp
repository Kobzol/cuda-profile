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

void RuntimeEmitter::store(Value* blockX,
                           Value* blockY,
                           Value* blockZ,
                           Value* threadX,
                           Value* threadY,
                           Value* threadZ,
                           Value* warpId,
                           Value* address,
                           Value* size,
                           Value* type
)
{
    this->builder.CreateCall(this->getStoreFunction(), {
            blockX, blockY, blockZ,
            threadX, threadY, threadZ,
            warpId, address, size,
            type
    });
}
void RuntimeEmitter::load(Value* blockX,
                          Value* blockY,
                          Value* blockZ,
                          Value* threadX,
                          Value* threadY,
                          Value* threadZ,
                          Value* warpId,
                          Value* address,
                          Value* size,
                          Value* type
)
{
    this->builder.CreateCall(this->getLoadFunction(), {
            blockX, blockY, blockZ,
            threadX, threadY, threadZ,
            warpId, address, size,
            type
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

void RuntimeEmitter::malloc(Value* address, Value* size, Value* type)
{
    this->builder.CreateCall(this->getMallocFunction(), {
            address, size, type
    });
}

void RuntimeEmitter::free(Value* address)
{
    this->builder.CreateCall(this->getFreeFunction(), {
            address
    });
}

llvm::Value* RuntimeEmitter::readInt32(const std::string& name)
{
    return this->builder.CreateCall(cast<Constant>(this->context.getModule()->getOrInsertFunction(
            name,
            this->context.getTypes().int32(),
            nullptr
    )));
}

Function* RuntimeEmitter::getStoreFunction()
{
    return cast<Function>(this->context.getModule()->getOrInsertFunction(
            prefix("store"),
            this->context.getTypes().voidType(),
            this->context.getTypes().int32(),
            this->context.getTypes().int32(),
            this->context.getTypes().int32(),
            this->context.getTypes().int32(),
            this->context.getTypes().int32(),
            this->context.getTypes().int32(),
            this->context.getTypes().int32(),
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
            this->context.getTypes().int32(),
            this->context.getTypes().int32(),
            this->context.getTypes().int32(),
            this->context.getTypes().int32(),
            this->context.getTypes().int32(),
            this->context.getTypes().int32(),
            this->context.getTypes().int32(),
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
