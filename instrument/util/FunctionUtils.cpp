#include "FunctionUtils.h"

#include <llvm/IR/Function.h>

using namespace llvm;

Instruction* FunctionUtils::getFirstInstruction(Function* function)
{
    for (auto& bb: function->getBasicBlockList())
    {
        for (auto& inst: bb.getInstList())
        {
            return &inst;
        }
    }

    return nullptr;
}
