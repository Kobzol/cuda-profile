#include "KernelLaunch.h"

#include <llvm/IR/Module.h>
#include <iostream>

#include "RuntimeEmitter.h"
#include "../util/Demangler.h"
#include "../util/Values.h"

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

    auto startEmitter = this->context.createEmitter(callSite);
    startEmitter.kernelStart();

    GlobalVariable* kernelNameCString = this->context.getValues().createGlobalCString(kernelName);

    auto endEmitter = this->context.createEmitter(callSite->getNextNode());
    endEmitter.kernelEnd(kernelNameCString);
}
