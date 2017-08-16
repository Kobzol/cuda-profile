#include "RuntimeEmitter.h"
#include "../util/Types.h"
#include "../util/Values.h"

#include <llvm/IR/Module.h>


using namespace llvm;

std::string prefix(const std::string& name)
{
    return "__cu_" + name;
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
    return this->builder.CreateCall(cast<Constant>(this->module->getOrInsertFunction(
            name,
            Types::int32(this->module),
            nullptr
    )));
}

Function* RuntimeEmitter::getStoreFunction()
{
    return cast<Function>(this->module->getOrInsertFunction(
            prefix("store"),
            Types::voidType(this->module),
            Types::int32(this->module),
            Types::int32(this->module),
            Types::int32(this->module),
            Types::int32(this->module),
            Types::int32(this->module),
            Types::int32(this->module),
            Types::int32(this->module),
            Types::voidPtr(this->module),
            Types::int64(this->module),
            Types::int8Ptr(this->module),
            nullptr));
}
llvm::Function* RuntimeEmitter::getLoadFunction()
{
    return cast<Function>(this->module->getOrInsertFunction(
            prefix("load"),
            Types::voidType(this->module),
            Types::int32(this->module),
            Types::int32(this->module),
            Types::int32(this->module),
            Types::int32(this->module),
            Types::int32(this->module),
            Types::int32(this->module),
            Types::int32(this->module),
            Types::voidPtr(this->module),
            Types::int64(this->module),
            Types::int8Ptr(this->module),
            nullptr));
}
Function* RuntimeEmitter::getKernelStartFunction()
{
    return cast<Function>(this->module->getOrInsertFunction(
            prefix("kernelStart"),
            Types::voidType(this->module),
            nullptr));
}
Function* RuntimeEmitter::getKernelEndFunction()
{
    return cast<Function>(this->module->getOrInsertFunction(
            prefix("kernelEnd"),
            Types::voidType(this->module),
            Types::int8Ptr(this->module),
            nullptr));
}
Function* RuntimeEmitter::getMallocFunction()
{
    return cast<Function>(this->module->getOrInsertFunction(
            prefix("malloc"),
            Types::voidType(this->module),
            Types::int8Ptr(this->module),
            Types::int64(this->module),
            Types::int8Ptr(this->module),
            nullptr));
}
Function* RuntimeEmitter::getFreeFunction()
{
    return cast<Function>(this->module->getOrInsertFunction(
            prefix("free"),
            Types::voidType(this->module),
            Types::int8Ptr(this->module),
            nullptr));
}
