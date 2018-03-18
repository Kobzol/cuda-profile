#include "Pass.h"

#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <iostream>

#include "emit/MemoryAccess.h"
#include "emit/KernelLaunch.h"
#include "emit/MemoryAlloc.h"
#include "../runtime/Prefix.h"
#include "emit/Kernel.h"
#include "emit/RuntimeEmitter.h"
#include "util/RegexFilter.h"
#include "Parameters.h"
#include "util/Demangler.h"

using namespace llvm;

namespace llvm {
#if __clang_major__ >= 5
    FunctionPass* createInferAddressSpacesPass();
#else
    FunctionPass* createNVPTXInferAddressSpacesPass();
#endif
    bool isKernelFunction(const Function& fn);
}

char CudaPass::ID = 0;

bool isInstrumentationFunction(Function& fn)
{
    return fn.getName().find(CU_PREFIX_STR) == 0;
}

CudaPass::CudaPass(): ModulePass(CudaPass::ID), filter((Parameters::kernelRegex()))
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
    RegexFilter filter(Parameters::kernelRegex());
    if (!this->isInstrumentableCuda(module))
    {
        return;
    }

    for (Function& fn : module.getFunctionList())
    {
        if (isKernelFunction(fn) && this->filter.matchesFunction(&fn))
        {
            std::cerr << "Instrumenting kernel " << Demangler().demangle(fn.getName().str()) << std::endl;
#if __clang_major__ >= 5
            auto pass = createInferAddressSpacesPass();
#else
            auto pass = createNVPTXInferAddressSpacesPass();
#endif
            pass->runOnFunction(fn);

            Kernel kernel(this->context);
            kernel.handleKernel(&fn);
        }
    }
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
        auto kernel = dyn_cast<Function>(call->getOperand(0)->stripPointerCasts());
        if (kernel && !this->filter.matchesFunction(kernel))
        {
            return;
        }

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
