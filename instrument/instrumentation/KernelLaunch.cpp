#include "KernelLaunch.h"
#include "../util/Types.h"
#include "RuntimeEmitter.h"

#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>

using namespace llvm;

void KernelLaunch::handleKernelLaunch(CallInst* callSite)
{
    RuntimeEmitter startEmitter(callSite);
    startEmitter.kernelStart();

    RuntimeEmitter endEmitter(callSite->getNextNode());
    endEmitter.kernelEnd();
}
