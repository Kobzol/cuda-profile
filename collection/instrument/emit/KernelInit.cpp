#include "KernelInit.h"
#include "../util/FunctionUtils.h"
#include "RuntimeEmitter.h"
#include "../util/DebugExtractor.h"

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

using namespace llvm;

static Type* getSharedBufferType(GlobalVariable* buffer)
{
    Type* type = buffer->getInitializer()->getType();

    if (auto arrType = dyn_cast<ArrayType>(type))
    {
        return arrType->getContainedType(0);
    }
    return type;
}
static std::string getSharedBufferName(GlobalVariable* buffer)
{
    DebugExtractor extractor;
    auto info = extractor.getDebugInfo(buffer);

    return info->hasName() ? info->getName() : "";
}

void KernelInit::handleKernelInit(Function* function, const std::vector<GlobalVariable*>& sharedBuffers)
{
    Module* module = function->getParent();
    RuntimeEmitter emitter(this->context, FunctionUtils::getFirstInstruction(function));
    auto sync = emitter.barrier();

    auto bufferBlock = BasicBlock::Create(module->getContext(), "sharedBuffers", function, sync->getParent());
    auto entryBlock = BasicBlock::Create(module->getContext(), "entry", function, bufferBlock);

    emitter.setInsertPoint(bufferBlock);
    for (auto& buffer: sharedBuffers)
    {
        size_t size, elementSize;
        this->context.getTypes().getGlobalVariableSize(buffer, size, elementSize);

        emitter.markSharedBuffer(
                emitter.getBuilder().CreatePointerCast(buffer, this->context.getTypes().voidPtr()),
                this->context.getValues().int64(size),
                this->context.getValues().int64(elementSize),
                this->context.getValues().int64(this->context.getTypeMapper().mapItem(getSharedBufferType(buffer))),
                this->context.getValues().int64(this->context.getNameMapper().mapItem(getSharedBufferName(buffer)))
        );
    }
    emitter.storeKernelDimensions();
    emitter.getBuilder().CreateBr(sync->getParent());

    emitter.setInsertPoint(entryBlock);
    emitter.getBuilder().CreateCondBr(emitter.isFirstThread(),
                               bufferBlock, sync->getParent());
}
