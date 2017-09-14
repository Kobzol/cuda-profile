#include "Pass.h"
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

using namespace llvm;
using namespace llvm::legacy;

static llvm::RegisterPass<CudaPass> Cuda("cu", "Cuda Pass", false, false);

static void registerMyPass(const PassManagerBuilder& builder, PassManagerBase& PM)
{
    PM.add(new CudaPass());
}
static RegisterStandardPasses RegisterMyPass(PassManagerBuilder::EP_EnabledOnOptLevel0, registerMyPass);
