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


void StoreHandler::handleKernel(Function* kernel)
{
    for (BasicBlock& bb : kernel->getBasicBlockList())
    {
        size_t index = 0;
        for (Instruction& inst : bb.getInstList())
        {
            if (auto* store = dyn_cast<StoreInst>(&inst))
            {
                if (index++ == 0)
                {
                    continue;
                }

                this->handleStore(store);
            }
            else if (auto* load = dyn_cast<LoadInst>(&inst))
            {
                this->handleLoad(load);
            }
        }
    }
}


void StoreHandler::handleStore(StoreInst* store)
{
    std::string type = this->context.getTypes().print(store->getValueOperand()->getType());
    auto* typeCString = this->context.getValues().createGlobalCString(type);

    auto emitter = this->context.createEmitter(store);
    emitter.store(emitter.readInt32(blockIdx("x")),
                  emitter.readInt32(blockIdx("y")),
                  emitter.readInt32(blockIdx("z")),
                  emitter.readInt32(threadIdx("x")),
                  emitter.readInt32(threadIdx("y")),
                  emitter.readInt32(threadIdx("z")),
                  emitter.readInt32(warpId()),
                  emitter.getBuilder().CreatePointerCast(store->getPointerOperand(), this->context.getTypes().voidPtr()),
                  this->context.getValues().int64(store->getValueOperand()->getType()->getPrimitiveSizeInBits() / 8),
                  typeCString
    );
}
void StoreHandler::handleLoad(LoadInst* load)
{
    std::string type = this->context.getTypes().print(load->getType());
    auto* typeCString = this->context.getValues().createGlobalCString(type);

    auto emitter = this->context.createEmitter(load);
    emitter.load(emitter.readInt32(blockIdx("x")),
                 emitter.readInt32(blockIdx("y")),
                 emitter.readInt32(blockIdx("z")),
                 emitter.readInt32(threadIdx("x")),
                 emitter.readInt32(threadIdx("y")),
                 emitter.readInt32(threadIdx("z")),
                 emitter.readInt32(warpId()),
                 emitter.getBuilder().CreatePointerCast(load->getPointerOperand(), this->context.getTypes().voidPtr()),
                 this->context.getValues().int64(load->getPointerOperand()->getType()->getPrimitiveSizeInBits() / 8),
                 typeCString
    );
}
