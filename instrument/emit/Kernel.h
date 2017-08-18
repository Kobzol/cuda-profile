#pragma once

#include <vector>

#include "../Context.h"
#include "../util/DebugInfo.h"

namespace llvm {
    class Function;
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
    void emitDebugInfo(llvm::Function* function, const std::vector<DebugInfo>& debugRecods);

    void instrumentStore(llvm::StoreInst* store, int32_t debugIndex);
    void instrumentLoad(llvm::LoadInst* load, int32_t debugIndex);

    bool isLocalStore(llvm::StoreInst* store);
    bool isLocalLoad(llvm::LoadInst* load);

    bool isInstrumentable(llvm::Instruction& instruction);
    void instrumentInstruction(llvm::Instruction* instruction, int32_t debugIndex);
};
