#include "Kernel.h"

#include <fstream>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/DebugInfoMetadata.h>

#include "MemoryAccess.h"
#include "RuntimeEmitter.h"
#include "../Parameters.h"
#include "../util/DebugExtractor.h"
#include "../util/Demangler.h"
#include "../util/LLVMAddressSpace.h"
#include "../util/FunctionUtils.h"
#include "../util/StringUtils.h"
#include "../util/FunctionContentLoader.h"
#include "KernelInit.h"

#define RAPIDJSON_HAS_STDSTRING 1
#include "../../runtime/format/json/rapidjson/prettywriter.h"
#include "../../runtime/format/json/rapidjson/ostreamwrapper.h"

using namespace llvm;
using namespace rapidjson;

static bool isLocalStore(StoreInst* store)
{
    return isa<AllocaInst>(store->getPointerOperand()->stripPointerCasts());
}
static bool isLocalLoad(LoadInst* load)
{
    return isa<AllocaInst>(load->getPointerOperand()->stripPointerCasts());
}

static bool isInstrumentable(Instruction& instruction)
{
    if (auto* store = dyn_cast<StoreInst>(&instruction))
    {
        // ignore parameter writes
        if (isa<Argument>(store->getValueOperand()))
        {
            return false;
        }

        if (!isLocalStore(store) || Parameters::shouldInstrumentLocals())
        {
            return true;
        }
    }
    else if (auto* load = dyn_cast<LoadInst>(&instruction))
    {
        if (!isLocalLoad(load) || Parameters::shouldInstrumentLocals())
        {
            return true;
        }
    }

    return false;
}

void Kernel::handleKernel(Function* function)
{
    auto sharedBuffers = this->extractSharedBuffers(function->getParent());
    this->emitFirstThreadActions(function, sharedBuffers);

    auto instructions = this->collectInstructions(function);
    auto debugRecords = this->instrumentInstructions(instructions);

    this->emitKernelMetadata(function, debugRecords, this->context.getTypeMapper(), this->context.getNameMapper());
}

std::vector<Instruction*> Kernel::collectInstructions(Function* function)
{
    std::vector<Instruction*> instructions;
    for (auto& block : function->getBasicBlockList())
    {
        for (auto& inst: block.getInstList())
        {
            if (isInstrumentable(inst))
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

void Kernel::emitKernelMetadata(Function* function, std::vector<DebugInfo> debugRecods,
                                Mapper<llvm::Type*>& typeMapper,
                                Mapper<std::string>& nameMapper)
{
    Demangler demangler;
    std::string name = demangler.demangle(function->getName().str());

    std::string kernelName = name.substr(0, name.find('('));
    DebugExtractor extractor;
    auto info = extractor.getDebugInfo(function);
    FunctionContentLoader loader;

    std::fstream metadataFile(kernelName + ".metadata.json", std::fstream::out);
    OStreamWrapper stream(metadataFile);
    PrettyWriter<OStreamWrapper> writer(stream);

    writer.StartObject();
    writer.String("type");
    writer.String("metadata");
    writer.String("kernel");
    writer.String(kernelName);

    writer.String("locations");
    writer.StartArray();
    for (auto& record: debugRecods)
    {
        writer.StartObject();
        writer.String("name");
        writer.String(record.getName());
        writer.String("file");
        writer.String(StringUtils::getFullPath(record.getFilename()));
        writer.String("line");
        writer.Int(record.getLine());
        writer.EndObject();
    }
    writer.EndArray();

    writer.String("typeMap");
    writer.StartArray();
    for (auto& item : typeMapper.getMappedItems())
    {
        writer.String(item);
    }
    writer.EndArray();

    writer.String("nameMap");
    writer.StartArray();
    for (auto& item : nameMapper.getMappedItems())
    {
        writer.String(item);
    }
    writer.EndArray();

    writer.String("source");
    writer.StartObject();
    writer.String("file");
    writer.String(info->isValid() ? StringUtils::getFullPath(info->getFilename()) : "");
    writer.String("line");
    writer.Int(info->isValid() ? info->getLine() : 0);
    writer.String("content");
    writer.String(info->isValid() ? loader.loadFunction(*info) : "");
    writer.EndObject();

    writer.EndObject();
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
    KernelInit init(this->context);
    init.handleKernelInit(function, sharedBuffers);
}
