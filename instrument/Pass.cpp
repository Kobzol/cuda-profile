#include "Pass.h"

#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <Target/NVPTX/NVPTXUtilities.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <iostream>

#include "emit/MemoryAccess.h"
#include "emit/KernelLaunch.h"
#include "emit/MemoryAlloc.h"
#include "../runtime/prefix.h"
#include "emit/Kernel.h"
#include "emit/RuntimeEmitter.h"

using namespace llvm;

namespace llvm {
    FunctionPass* createNVPTXInferAddressSpacesPass();
}

char CudaPass::ID = 0;

bool isInstrumentationFunction(Function& fn)
{
    return fn.getName().find(PREFIX_STR) == 0;
}

CudaPass::CudaPass(): ModulePass(CudaPass::ID)
{

}

bool CudaPass::runOnModule(Module& module)
{
    std::cerr << "Run on module: " << module.getName().str() << " " << module.getTargetTriple() << std::endl;

    this->context.setModule(&module);

    bool cuda = module.getTargetTriple() == "nvptx64-nvidia-cuda";
    cuda ? this->instrumentCuda(module) : this->instrumentCpp(module);

    return false;
}

void CudaPass::instrumentCuda(Module& module)
{
    if (!this->isInstrumentableCuda(module))
    {
        return;
    }

    for (Function& fn : module.getFunctionList())
    {
        if (isKernelFunction(fn))
        {
            auto pass = createNVPTXInferAddressSpacesPass();
            pass->runOnFunction(fn);
            Kernel kernel(this->context);
            kernel.handleKernel(&fn);
        }
    }
}

Function* CudaPass::augmentKernel(Function* fn)
{
    if (this->kernelMap.find(fn) == this->kernelMap.end())
    {
        FunctionType *type = fn->getFunctionType();
        std::vector<Type *> arguments;
        arguments.insert(arguments.end(), type->param_begin(), type->param_end());
        arguments.push_back(this->context.getTypes().int32());

        FunctionType *newType = FunctionType::get(type->getReturnType(), arguments, type->isVarArg());

        Function *augmented = Function::Create(newType, fn->getLinkage(), fn->getName().str() + "_clone");
        fn->getParent()->getFunctionList().push_back(augmented);

        ValueToValueMapTy map;

        auto newArgs = augmented->arg_begin();
        for (auto args = fn->arg_begin(); args != fn->arg_end(); ++args, ++newArgs) {
            map[&(*args)] = &(*newArgs);
            newArgs->setName(args->getName());
        }

        SmallVector<ReturnInst *, 100> returns;
        CloneFunctionInto(augmented, fn, map, true, returns);
        // TODO: CloneDebugInfoMetadata

        augmented->setCallingConv(fn->getCallingConv());

        this->kernelMap[fn] = augmented;
        return augmented;
    }

    return this->kernelMap[fn];
}

void CudaPass::instrumentCpp(Module& module)
{
    if (!this->isInstrumentableCpp(module))
    {
        return;
    }

    for (Function& fn : module.getFunctionList())
    {
        if (!isInstrumentationFunction(fn))
        {
            for (BasicBlock& bb : fn.getBasicBlockList())
            {
                for (Instruction& inst : bb.getInstList())
                {
                    if (auto* call = dyn_cast<CallInst>(&inst))
                    {
                        auto calledFn = call->getCalledFunction();
                        if (calledFn != nullptr)
                        {
                            this->handleFunctionCall(call);
                        }
                    }
                }
            }
        }
    }
}

void CudaPass::handleFunctionCall(CallInst* call)
{
    Function* calledFn = call->getCalledFunction();
    auto name = calledFn->getName();

    if (name == "cudaLaunch")
    {
        KernelLaunch kernelLaunch(this->context);
        kernelLaunch.handleKernelLaunch(call);
    }
    else if (name == "cudaMalloc")
    {
        MemoryAlloc memoryAlloc(this->context);
        memoryAlloc.handleCudaMalloc(call);
    }
    else if (name == "cudaFree")
    {
        MemoryAlloc memoryAlloc(this->context);
        memoryAlloc.handleCudaFree(call);
    }
}

bool CudaPass::isInstrumentableCuda(Module& module)
{
    return module.getFunction(RuntimeEmitter::runtimePrefix("store")) != nullptr;
}
bool CudaPass::isInstrumentableCpp(Module& module)
{
    return module.getFunction(RuntimeEmitter::runtimePrefix("kernelStart")) != nullptr;
}
