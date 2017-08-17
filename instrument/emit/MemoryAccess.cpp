#include "MemoryAccess.h"

#include <iostream>
#include <llvm/IR/Module.h>
#include <llvm/IR/TypeFinder.h>
#include <llvm/Support/raw_ostream.h>

#include "../util/Types.h"
#include "../util/Values.h"
#include "RuntimeEmitter.h"

using namespace llvm;


std::string threadIdx(const std::string& dim)
{
    return "llvm.nvvm.read.ptx.sreg.tid." + dim;
}
std::string blockIdx(const std::string& dim)
{
    return "llvm.nvvm.read.ptx.sreg.ctaid." + dim;
}
std::string warpId()
{
    return "llvm.nvvm.read.ptx.sreg.warpid";
}


void MemoryAccess::handleStore(StoreInst* store, size_t debugIndex)
{
    std::string type = this->context.getTypes().print(store->getValueOperand()->getType());
    auto* typeCString = this->context.getValues().createGlobalCString(type);

    auto emitter = this->context.createEmitter(store);
    emitter.store(emitter.getBuilder().CreatePointerCast(store->getPointerOperand(), this->context.getTypes().voidPtr()),
                  this->context.getValues().int64(store->getValueOperand()->getType()->getPrimitiveSizeInBits() / 8),
                  typeCString,
                  this->context.getValues().int32(debugIndex)
    );
}
void MemoryAccess::handleLoad(LoadInst* load, size_t debugIndex)
{
    std::string type = this->context.getTypes().print(load->getType());
    auto* typeCString = this->context.getValues().createGlobalCString(type);

    auto emitter = this->context.createEmitter(load);
    emitter.load(emitter.getBuilder().CreatePointerCast(load->getPointerOperand(), this->context.getTypes().voidPtr()),
                 this->context.getValues().int64(load->getPointerOperand()->getType()->getPrimitiveSizeInBits() / 8),
                 typeCString,
                 this->context.getValues().int32(debugIndex)
    );
}
