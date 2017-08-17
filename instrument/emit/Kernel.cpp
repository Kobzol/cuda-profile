#include "Kernel.h"
#include "MemoryAccess.h"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

using namespace llvm;


void Kernel::handleKernel(Function* function)
{
    this->traverseInstructions(function);
}

void Kernel::handleStore(StoreInst* store)
{
    MemoryAccess handler(this->context);
    handler.handleStore(store);
}

void Kernel::handleLoad(LoadInst* load)
{
    MemoryAccess handler(this->context);
    handler.handleLoad(load);
}

void Kernel::traverseInstructions(Function* function)
{
    for (BasicBlock& bb : function->getBasicBlockList())
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
