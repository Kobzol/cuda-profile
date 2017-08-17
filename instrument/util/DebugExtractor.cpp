#include "DebugExtractor.h"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>

using namespace llvm;


DebugInfo* DebugExtractor::getDebugInfo(Value* inst)
{
    return this->getFromCache(inst);
}
DebugInfo DebugExtractor::getInstructionLocation(Instruction* inst)
{
    auto& debugLoc = inst->getDebugLoc();
    DILocation* loc = debugLoc.get();

    if (loc == nullptr)
    {
        return DebugInfo();
    }

    return DebugInfo(inst->getOpcodeName(), loc->getFilename().str(), debugLoc.getLine());
}

const MDNode* DebugExtractor::findVarInFunction(const Function* function, const Value* value)
{
    for (auto& block : function->getBasicBlockList())
    {
        for (auto& inst : block.getInstList())
        {
            if (const auto* DbgDeclare = dyn_cast<DbgDeclareInst>(&inst))
            {
                if (DbgDeclare->getAddress() == value)
                {
                    return DbgDeclare->getVariable();
                }
            }
            else if (const auto* DbgValue = dyn_cast<DbgValueInst>(&inst))
            {
                if (DbgValue->getValue() == value)
                {
                    return DbgValue->getVariable();
                }
            }
        }
    }

    return nullptr;
}

const Function* DebugExtractor::findVarScope(const Value* value)
{
    if (const auto* Arg = dyn_cast<Argument>(value))
    {
        return Arg->getParent();
    }
    if (const auto* I = dyn_cast<Instruction>(value))
    {
        return I->getParent()->getParent();
    }

    return nullptr;
}

std::unique_ptr<DebugInfo> DebugExtractor::getGlobalVarDebugInfo(const GlobalVariable* global)
{
    const Module* module = global->getParent();
    if (NamedMDNode* cu = module->getNamedMetadata("llvm.dbg.cu"))
    {
        for (unsigned i = 0, e = cu->getNumOperands(); i != e; ++i)
        {
            auto* compileUnit = cast<DICompileUnit>(cu->getOperand(i));
            for (DIGlobalVariableExpression* globalDI : compileUnit->getGlobalVariables())
            {
                /*if (globalDI->getVariable() == global)
                {
                    return std::make_unique<DebugInfo>(globalDI);
                }*/
            }
        }
    }

    return std::make_unique<DebugInfo>();
}

std::unique_ptr<DebugInfo> DebugExtractor::getVarDebugInfo(const Value* value)
{
    if (const auto* global = dyn_cast<GlobalVariable>(value))
    {
        return this->getGlobalVarDebugInfo(global);
    }

    const Function* function = this->findVarScope(value);
    if (function == nullptr)
    {
        return std::make_unique<DebugInfo>(value->getName().str());
    }

    const MDNode* debugVar = this->findVarInFunction(function, value);
    if (debugVar == nullptr)
    {
        return std::make_unique<DebugInfo>();
    }

    return std::make_unique<DebugInfo>(dyn_cast<DIVariable>(debugVar));
}

DebugInfo* DebugExtractor::getFromCache(const Value* value)
{
    if (this->debugCache.find(value) == this->debugCache.end())
    {
        this->debugCache[value] = this->getVarDebugInfo(value);
    }

    return this->debugCache.at(value).get();
}
