#include "KernelLaunch.h"

#include <llvm/IR/Module.h>
#include <iostream>

#include "RuntimeEmitter.h"
#include "../util/Demangler.h"

using namespace llvm;


std::string getKernelName(CallInst* callSite)
{
    Value* kernel = callSite->getOperand(0)->stripPointerCasts();
    if (auto* function = dyn_cast<Function>(kernel))
    {
        auto name = Demangler().demangle(function->getName().str());
        return name.substr(0, name.find('('));
    }

    return "kernel@" + std::to_string((size_t) kernel);
}

void KernelLaunch::handleKernelLaunch(CallInst* callSite)
{
    auto kernelName = getKernelName(callSite);

    RuntimeEmitter startEmitter(callSite);
    startEmitter.kernelStart();

    RuntimeEmitter endEmitter(callSite->getNextNode());
    endEmitter.kernelEnd(kernelName);
}
