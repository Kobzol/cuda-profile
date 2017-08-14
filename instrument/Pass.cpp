#include "Pass.h"
#include "Store.h"

#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <Target/NVPTX/NVPTXUtilities.h>
#include <iostream>

using namespace llvm;

char CudaPass::ID = 0;

CudaPass::CudaPass(): ModulePass(CudaPass::ID)
{

}

bool CudaPass::runOnModule(Module& module)
{
    std::cerr << "Run on module: " << module.getName().str() << " " << module.getTargetTriple() << std::endl;

    bool cuda = module.getTargetTriple() == "nvptx64-nvidia-cuda";
    cuda ? this->instrumentCuda(module) : this->instrumentCpp(module);

    return false;
}

void CudaPass::instrumentCuda(Module& module)
{
    for (Function& fn : module.getFunctionList())
    {
        if (isKernelFunction(fn))
        {
            StoreHandler handler;
            handler.handleKernel(&fn);

            fn.dump();
        }
    }
}

void CudaPass::instrumentCpp(Module& module)
{
    return;
    for (Function& fn : module.getFunctionList())
    {
        for (BasicBlock& bb : fn.getBasicBlockList())
        {
            for (Instruction& inst : bb.getInstList())
            {
                if (auto* call = dyn_cast<CallInst>(&inst))
                {
                    auto calledName = call->getCalledFunction()->getName().str();
                    if (calledName.find("kernel") != std::string::npos && calledName.find("__cu") == std::string::npos)
                    {
                        auto* fnCall = cast<Function>(module.getOrInsertFunction("__cu_kernelEnd",
                                                                                   Type::getVoidTy(module.getContext()),
                                                                                   nullptr));

                        IRBuilder<> builder(call->getNextNode());
                        builder.CreateCall(fnCall, {

                        });
                    }
                }
            }
        }
    }
}
