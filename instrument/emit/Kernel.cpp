#include "Kernel.h"

#include <fstream>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

#include "MemoryAccess.h"
#include "../util/DebugExtractor.h"
#include "../util/Demangler.h"
#include "../../runtime/picojson.h"
#include "../util/StringUtils.h"

using namespace llvm;


void Kernel::handleKernel(Function* function)
{
    auto instructions = this->collectInstructions(function);
    auto debugRecords = this->instrumentInstructions(instructions);
    this->emitDebugInfo(function, debugRecords);
}

std::vector<Instruction*> Kernel::collectInstructions(Function* function)
{
    std::vector<Instruction*> instructions;
    for (auto& block : function->getBasicBlockList())
    {
        for (auto& inst: block.getInstList())
        {
            if (this->isInstrumentable(inst))
            {
                instructions.push_back(&inst);
            }
        }
    }

    return instructions;
}
std::vector<DebugInfo> Kernel::instrumentInstructions(const std::vector<Instruction*>& instructions)
{
    DebugExtractor extractor;
    std::vector<DebugInfo> debugInfo;

    for (auto* inst : instructions)
    {
        DebugInfo info = extractor.getInstructionLocation(inst);
        size_t debugIndex = 0;
        if (info.isValid())
        {
            debugInfo.push_back(info);
            debugIndex++;
        }

        this->instrumentInstruction(inst, debugIndex);
    }

    return debugInfo;
}

bool Kernel::isLocalStore(StoreInst* store)
{
    return isa<AllocaInst>(store->getPointerOperand());
}
bool Kernel::isLocalLoad(LoadInst* load)
{
    return isa<AllocaInst>(load->getPointerOperand());
}

bool Kernel::isInstrumentable(Instruction& instruction)
{
    if (auto* store = dyn_cast<StoreInst>(&instruction))
    {
        if (!this->isLocalStore(store))
        {
            return true;
        }
    }
    else if (auto* load = dyn_cast<LoadInst>(&instruction))
    {
        if (!this->isLocalLoad(load))
        {
            return true;
        }
    }

    return false;
}

void Kernel::instrumentInstruction(Instruction* instruction, size_t debugIndex)
{
    if (auto* store = dyn_cast<StoreInst>(instruction))
    {
        this->instrumentStore(store, debugIndex);
    }
    else if (auto* load = dyn_cast<LoadInst>(instruction))
    {
        this->instrumentLoad(load, debugIndex);
    }
}

void Kernel::instrumentStore(StoreInst* store, size_t debugIndex)
{
    MemoryAccess handler(this->context);
    handler.handleStore(store, debugIndex);
}

void Kernel::instrumentLoad(LoadInst* load, size_t debugIndex)
{
    MemoryAccess handler(this->context);
    handler.handleLoad(load, debugIndex);
}

void Kernel::emitDebugInfo(Function* function, const std::vector<DebugInfo>& debugRecods)
{
    Demangler demangler;
    std::string name = demangler.demangle(function->getName().str());

    std::vector<picojson::value> jsonRecords;
    for (auto& info : debugRecods)
    {
        jsonRecords.push_back(picojson::value(picojson::object {
                {"name", picojson::value(info.getName())},
                {"file", picojson::value(StringUtils::getFullPath(info.getFilename()))},
                {"line", picojson::value((double) info.getLine())}
        }));
    }

    std::fstream debugFile("debug-" + name.substr(0, name.find('(')) + ".json", std::fstream::out);
    debugFile << picojson::value(picojson::array {
        jsonRecords
    }).serialize(true);
}
