#pragma once

#include <llvm/IR/Instructions.h>

namespace llvm {
    class StoreInst;
    class Function;
    class Module;
}

class StoreHandler
{
public:
    void handleKernel(llvm::Function* kernel);

private:
    void handleStore(llvm::StoreInst* store);
};
