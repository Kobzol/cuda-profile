#include "Values.h"
#include "Types.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>

using namespace llvm;

Value* Values::int32(Module* module, int32_t value)
{
    return ConstantInt::get(Types::int32(module), value, false);
}
Value* Values::int64(Module* module, int64_t value)
{
    return ConstantInt::get(Types::int64(module), value, false);
}

GlobalVariable* Values::createGlobalCString(Module* module, const std::string& name, const std::string& value)
{
    GlobalVariable* global = module->getGlobalVariable(name, true);

    if (global == nullptr)
    {
        Constant* nameCString = ConstantDataArray::getString(module->getContext(), value, true);
        Type* stringType = ArrayType::get(Types::int8(module), value.size() + 1);

        global = static_cast<GlobalVariable*>(module->getOrInsertGlobal(name, stringType));
        global->setLinkage(GlobalValue::LinkageTypes::PrivateLinkage);
        global->setConstant(true);
        global->setInitializer(nameCString);
    }

    return global;
}
