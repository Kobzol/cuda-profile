#pragma once

namespace llvm {
    class Function;
    class Instruction;
}

class FunctionUtils
{
public:
    static llvm::Instruction* getFirstInstruction(llvm::Function* function);
};
