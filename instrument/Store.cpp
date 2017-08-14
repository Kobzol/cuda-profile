#include "Store.h"
#include "util/Types.h"
#include "util/Values.h"

#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <iostream>

using namespace llvm;

template <typename T, typename R>
Value* readInt(IRBuilder<T, R>& builder, Module* module, const std::string& name)
{
    return builder.CreateCall(cast<Constant>(module->getOrInsertFunction(name,
                                                             Types::int32(module),
                                                             nullptr
    )));
}
std::string threadIdx(const std::string& dim)
{
    return "llvm.nvvm.read.ptx.sreg.tid." + dim;
}
std::string blockIdx(const std::string& dim)
{
    return "llvm.nvvm.read.ptx.sreg.ctaid." + dim;
}


void StoreHandler::handleKernel(Function* kernel)
{
    for (BasicBlock& bb : kernel->getBasicBlockList())
    {
        size_t index = 0;
        for (Instruction& inst : bb.getInstList())
        {
            if (auto* store = dyn_cast<StoreInst>(&inst))
            {
                if (index++ == 0)
                {
                    continue;
                }

                this->handleStore(store);
            }
        }
    }
}


void StoreHandler::handleStore(StoreInst* store)
{
    std::cerr << store->getPointerAddressSpace() << std::endl;
    Module* module = store->getModule();

    IRBuilder<> builder(store);
    builder.CreateCall(this->getStoreFunction(module), {
            readInt(builder, module, blockIdx("x")),
            readInt(builder, module, blockIdx("y")),
            readInt(builder, module, blockIdx("z")),
            readInt(builder, module, threadIdx("x")),
            readInt(builder, module, threadIdx("y")),
            readInt(builder, module, threadIdx("z")),
            builder.CreatePointerCast(store->getPointerOperand(), Types::voidPtr(module)),
            Values::int64(module, store->getValueOperand()->getType()->getPrimitiveSizeInBits() / 8)
    });
}

Function* StoreHandler::getStoreFunction(Module* module)
{
    return cast<Function>(module->getOrInsertFunction("__cu_store",
                                                      Types::voidType(module),
                                                      Types::int32(module),
                                                      Types::int32(module),
                                                      Types::int32(module),
                                                      Types::int32(module),
                                                      Types::int32(module),
                                                      Types::int32(module),
                                                      Types::voidPtr(module),
                                                      Types::int64(module),
                                                      nullptr));
}
