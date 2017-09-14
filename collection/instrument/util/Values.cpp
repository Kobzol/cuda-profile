#include "Values.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>

using namespace llvm;

Value* Values::int32(int32_t value)
{
    return ConstantInt::get(this->types.int32(), value, false);
}
Value* Values::int64(int64_t value)
{
    return ConstantInt::get(this->types.int64(), value, false);
}

GlobalVariable* Values::createGlobalCString(const std::string& value)
{
    if (this->cStringMap.find(value) == this->cStringMap.end())
    {
        Constant* nameCString = ConstantDataArray::getString(this->module->getContext(), value, true);
        Type* stringType = ArrayType::get(this->types.int8(), value.size() + 1);

        auto* global = dyn_cast<GlobalVariable>(this->module->getOrInsertGlobal(this->generateGlobalName(), stringType));
        global->setLinkage(GlobalValue::LinkageTypes::PrivateLinkage);
        global->setConstant(true);
        global->setInitializer(nameCString);

        this->cStringMap[value] = global;
        return global;
    }

    return this->cStringMap[value];
}

std::string Values::generateGlobalName()
{
    while (true)
    {
        std::string name = "__cu_ProfileGlobal_" + std::to_string(this->globalCounter++);
        GlobalVariable* global = module->getGlobalVariable(name, true);

        if (global == nullptr)
        {
            return name;
        }
    }
}
