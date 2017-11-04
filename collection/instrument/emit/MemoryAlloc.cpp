#include "MemoryAlloc.h"

#include "RuntimeEmitter.h"
#include "../util/DebugExtractor.h"
#include "../util/StringUtils.h"

using namespace llvm;

static DebugInfo getCudaMallocDebug(CallInst* malloc)
{
    DebugExtractor extractor;
    auto varInfo = extractor.getDebugInfo(malloc->getOperand(0)->stripPointerCasts());
    auto callInfo = extractor.getInstructionLocation(malloc);

    return DebugInfo(varInfo->getName(), callInfo.getFilename(), callInfo.getLine());
}
static Type* getMallocValueType(CallInst* malloc)
{
    // pass through two levels of indirection
    auto* bufferPtrType = dyn_cast<PointerType>(malloc->getOperand(0)->stripPointerCasts()->getType());
    auto* valuePtrType = dyn_cast<PointerType>(bufferPtrType->getElementType());

    return valuePtrType->getElementType();
}
static std::string getFullPath(const DebugInfo& info)
{
    if (info.getFilename().empty()) return "";

    return StringUtils::getFullPath(info.getFilename()) + ":" + std::to_string(info.getLine());
}

void MemoryAlloc::handleCudaMalloc(CallInst* call)
{
    auto emitter = this->context.createEmitter(call->getNextNode());

    Value* addressLoad = emitter.getBuilder().CreateLoad(call->getOperand(0));

    Type* valueType = getMallocValueType(call);
    std::string typeStr = this->context.getTypes().stringify(valueType);
    GlobalVariable* typeCString = this->context.getValues().createGlobalCString(typeStr);

    auto debug = getCudaMallocDebug(call);
    emitter.malloc(
            addressLoad,
            call->getOperand(1),
            this->context.getValues().int64(valueType->getPrimitiveSizeInBits() / 8),
            typeCString,
            this->context.getValues().createGlobalCString(debug.getName()),
            this->context.getValues().createGlobalCString(getFullPath(debug))
    );
}

void MemoryAlloc::handleCudaFree(CallInst* call)
{
    auto emitter = this->context.createEmitter(call);
    emitter.free(call->getOperand(0));
}
