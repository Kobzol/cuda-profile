#pragma once

namespace llvm {
    class Function;
    class LoadInst;
    class Module;
    class StoreInst;
}

class StoreHandler
{
public:
    void handleKernel(llvm::Function* kernel);

private:
    void handleStore(llvm::StoreInst* store);
    void handleLoad(llvm::LoadInst* load);
};
