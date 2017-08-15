#include "Store.h"
#include "../util/Types.h"
#include "../util/Values.h"
#include "RuntimeEmitter.h"

#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <iostream>

using namespace llvm;

std::string threadIdx(const std::string& dim)
{
    return "llvm.nvvm.read.ptx.sreg.tid." + dim;
}
std::string blockIdx(const std::string& dim)
{
    return "llvm.nvvm.read.ptx.sreg.ctaid." + dim;
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
        }
    }
}


void StoreHandler::handleStore(StoreInst* store)
{
    RuntimeEmitter emitter(store);
    emitter.store(emitter.readInt32(blockIdx("x")),
                  emitter.readInt32(blockIdx("y")),
                  emitter.readInt32(blockIdx("z")),
                  emitter.readInt32(threadIdx("x")),
                  emitter.readInt32(threadIdx("y")),
                  emitter.readInt32(threadIdx("z")),
                  emitter.getBuilder().CreatePointerCast(store->getPointerOperand(), Types::voidPtr(store->getModule())),
                  Values::int64(store->getModule(), store->getValueOperand()->getType()->getPrimitiveSizeInBits() / 8));
}
