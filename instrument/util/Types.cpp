#include "Types.h"
#include "StringUtils.h"

#include <llvm/IR/Module.h>
#include <llvm/IR/TypeFinder.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>

using namespace llvm;


Type* Types::voidType()
{
    return Type::getVoidTy(this->module->getContext());
}

Type* Types::int8()
{
    return Type::getInt8Ty(this->module->getContext());
}
llvm::Type *Types::int32()
{
    return Type::getInt32Ty(this->module->getContext());
}
Type* Types::int64()
{
    return Type::getInt64Ty(this->module->getContext());
}
Type* Types::boolType()
{
    return Type::getInt1Ty(this->module->getContext());
}

PointerType* Types::voidPtr()
{
    return this->int8Ptr();
}
PointerType* Types::int8Ptr()
{
    return this->int8()->getPointerTo();
}
PointerType* Types::int32Ptr()
{
    return this->int32()->getPointerTo();
}
PointerType* Types::int64Ptr()
{
    return this->int64()->getPointerTo();
}

StructType* Types::getCompositeType(const std::string& name)
{
    if (this->structMap.find(name) == this->structMap.end())
    {
        TypeFinder structFinder;
        structFinder.run(*this->module, true);
        for (auto* structType : structFinder)
        {
            if (structType->getStructName() == "struct." + name ||
                structType->getStructName() == "class." + name)
            {
                this->structMap[name] = structType;
                return structType;
            }
        }

        return nullptr;
    }

    return this->structMap[name];
}

std::string Types::print(Type* type)
{
    if (isa<StructType>(type))
    {
        return StringUtils::trimStart(
                StringUtils::trimStart(
                        type->getStructName().str(),
                        "class."),
                "struct.");
    }

    std::string buffer;
    llvm::raw_string_ostream stream(buffer);
    type->print(stream, false, true);

    return stream.str();
}
