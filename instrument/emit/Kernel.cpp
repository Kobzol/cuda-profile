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
        for (Instruction& inst : bb.getInstList())
        {
            if (auto* store = dyn_cast<StoreInst>(&inst))
            {
                if (!this->isLocalStore(store))
                {
                    this->handleStore(store);
                }
            }
            else if (auto* load = dyn_cast<LoadInst>(&inst))
            {
                if (!this->isLocalLoad(load))
                {
                    this->handleLoad(load);
                }
            }
        }
    }
}

bool Kernel::isLocalStore(StoreInst* store)
{
    return isa<AllocaInst>(store->getPointerOperand());
}
bool Kernel::isLocalLoad(LoadInst* load)
{
    return isa<AllocaInst>(load->getPointerOperand());
}
