#include "Kernel.h"

#include <fstream>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/DebugInfoMetadata.h>

#include "MemoryAccess.h"
#include "../util/DebugExtractor.h"
#include "../util/Demangler.h"
#include "../../runtime/format/json/picojson.h"
#include "../util/LLVMAddressSpace.h"
#include "../util/FunctionUtils.h"
#include "../util/StringUtils.h"
#include "RuntimeEmitter.h"
#include "../util/FunctionContentLoader.h"

using namespace llvm;

void Kernel::handleKernel(Function* function)
{
    auto sharedBuffers = this->extractSharedBuffers(function->getParent());
    this->emitFirstThreadActions(function, sharedBuffers);

    auto instructions = this->collectInstructions(function);
    auto debugRecords = this->instrumentInstructions(instructions);

    this->emitKernelMetadata(function, debugRecords, this->context.getTypeMapper());
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

    int32_t debugIndex = 0;
    for (auto* inst : instructions)
    {
        DebugInfo info = extractor.getInstructionLocation(inst);
        int32_t index = -1;

        if (info.isValid())
        {
            debugInfo.push_back(info);
            index = debugIndex;
            debugIndex++;
        }

        this->instrumentInstruction(inst, index);
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

void Kernel::instrumentInstruction(Instruction* instruction, int32_t debugIndex)
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

void Kernel::instrumentStore(StoreInst* store, int32_t debugIndex)
{
    MemoryAccess handler(this->context);
    handler.handleStore(store, debugIndex);
}

void Kernel::instrumentLoad(LoadInst* load, int32_t debugIndex)
{
    MemoryAccess handler(this->context);
    handler.handleLoad(load, debugIndex);
}

void Kernel::emitKernelMetadata(Function* function, std::vector<DebugInfo> debugRecods, TypeMapper& mapper)
{
    Demangler demangler;
    std::string name = demangler.demangle(function->getName().str());

    std::vector<picojson::value> jsonRecords;
    for (auto& info: debugRecods)
    {
        jsonRecords.push_back(picojson::value(picojson::object {
                {"name", picojson::value(info.getName())},
                {"file", picojson::value(StringUtils::getFullPath(info.getFilename()))},
                {"line", picojson::value((double) info.getLine())}
        }));
    }
    std::vector<picojson::value> jsonTypes;
    for (auto& type : mapper.getTypes())
    {
        jsonTypes.emplace_back(type);
    }

    std::string kernelName = name.substr(0, name.find('('));
    DebugExtractor extractor;
    auto info = extractor.getDebugInfo(function);
    FunctionContentLoader loader;

    std::fstream metadataFile(kernelName + ".metadata.json", std::fstream::out);
    metadataFile << picojson::value(picojson::object {
            {"type", picojson::value("metadata")},
            {"kernel", picojson::value(kernelName)},
            {"locations", picojson::value(jsonRecords)},
            {"typeMap", picojson::value(jsonTypes)},
            {"source", picojson::value(picojson::object {
                    {"file", picojson::value(info->isValid() ? StringUtils::getFullPath(info->getFilename()) : "")},
                    {"line", picojson::value((double) (info->isValid() ? info->getLine() : 0))},
                    {"content", picojson::value(info->isValid() ? loader.loadFunction(*info) : "")}
            })}
    }).serialize(true);
}

std::vector<GlobalVariable*> Kernel::extractSharedBuffers(Module* module)
{
    std::vector<GlobalVariable*> sharedBuffers;

    for (auto& glob: module->getGlobalList())
    {
        if (this->isSharedBuffer(glob))
        {
            sharedBuffers.push_back(&glob);
        }
    }

    return sharedBuffers;
}

bool Kernel::isSharedBuffer(GlobalVariable& variable)
{
    return static_cast<LLVMAddressSpace>(variable.getType()->getAddressSpace()) == LLVMAddressSpace::Shared;
}

void Kernel::emitFirstThreadActions(Function* function, const std::vector<GlobalVariable*>& sharedBuffers)
{
    RuntimeEmitter emitter(this->context, FunctionUtils::getFirstInstruction(function));
    emitter.emitFirstThreadActions(sharedBuffers);
}
