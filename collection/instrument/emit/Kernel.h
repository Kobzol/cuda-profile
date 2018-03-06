#pragma once

#include <vector>

#include "../Context.h"
#include "../util/DebugInfo.h"

namespace llvm {
    class Function;
    class GlobalVariable;
    class Instruction;
    class LoadInst;
    class StoreInst;
}


class Kernel
{
public:
    explicit Kernel(Context& context): context(context)
    {

    }

    void handleKernel(llvm::Function* function);

private:
    Context& context;

    std::vector<llvm::Instruction*> collectInstructions(llvm::Function* function);
    std::vector<DebugInfo> instrumentInstructions(const std::vector<llvm::Instruction*>& instructions);
    void emitKernelMetadata(llvm::Function* function, std::vector<DebugInfo> debugRecods, TypeMapper& mapper);

    void instrumentStore(llvm::StoreInst* store, int32_t debugIndex);
    void instrumentLoad(llvm::LoadInst* load, int32_t debugIndex);
    void instrumentInstruction(llvm::Instruction* instruction, int32_t debugIndex);

    std::vector<llvm::GlobalVariable*> extractSharedBuffers(llvm::Module* module);
    bool isSharedBuffer(llvm::GlobalVariable& variable);

    void emitFirstThreadActions(llvm::Function* function, const std::vector<llvm::GlobalVariable*>& sharedBuffers);
};
