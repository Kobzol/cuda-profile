#include "Pass.h"
#include <iostream>

#include <llvm/IR/Module.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>

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
        std::string name = fn.getName().str();
        if (name.find("__cu_") != std::string::npos)
        {
            continue;
        }

        for (BasicBlock& bb : fn.getBasicBlockList())
        {
            for (Instruction& inst : bb.getInstList())
            {
                if (auto* store = dyn_cast<StoreInst>(&inst))
                {
                    Type* typeInt32 = Type::getInt32Ty(module.getContext());
                    PointerType* typeInt8ptr = Type::getInt8Ty(module.getContext())->getPointerTo();

                    Function* fnCall = cast<Function>(module.getOrInsertFunction("__cu_store",
                                                                                 Type::getVoidTy(module.getContext()),
                                                                                 typeInt32,
                                                                                 typeInt8ptr,
                                                                                 nullptr));
                    Constant* fnTid = cast<Constant>(module.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.tid.x",
                                                                                typeInt32,
                                                                                nullptr
                    ));

                    IRBuilder<> builder(store);
                    builder.CreateCall(fnCall, {
                            builder.CreateCall(fnTid),
                            ConstantPointerNull::get(typeInt8ptr)
                    });
                }
            }
        }

        fn.dump();
    }
}

void CudaPass::instrumentCpp(Module& module)
{
    /*for (Function& fn : module.getFunctionList())
    {
        for (BasicBlock& bb : fn.getBasicBlockList())
        {
            for (Instruction& inst : bb.getInstList())
            {
                if (auto* call = dyn_cast<CallInst>(&inst))
                {
                    auto calledName = call->getCalledFunction()->getName().str();
                    if (calledName.find("kernel") != std::string::npos)
                    {
                        Function* fnCall = cast<Function>(module.getOrInsertFunction("__cu_pullDevice",
                                                                                   Type::getVoidTy(module.getContext()),
                                                                                   nullptr));

                        IRBuilder<> builder(call->getNextNode());
                        builder.CreateCall(fnCall, {

                        });
                    }
                }
            }
        }

        fn.dump();
    }*/
}
