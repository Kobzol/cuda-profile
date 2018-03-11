#include "Pass.h"
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

using namespace llvm;
using namespace llvm::legacy;

static llvm::RegisterPass<CudaPass> Cuda("cu", "Cuda profiler pass", false, false);
static bool PassRegistered = false;

static void registerMyPass(const PassManagerBuilder& builder, PassManagerBase& PM)
{
    if (!PassRegistered)
    {
        PM.add(new CudaPass());
        PassRegistered = true;
    }
}
static RegisterStandardPasses RegisterMyPassDebug(PassManagerBuilder::EP_EnabledOnOptLevel0, registerMyPass);
static RegisterStandardPasses RegisterMyPassOpt(PassManagerBuilder::EP_ModuleOptimizerEarly, registerMyPass);
