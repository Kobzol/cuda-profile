#include "Types.h"
#include "StringUtils.h"

#include <llvm/IR/Module.h>
#include <llvm/IR/TypeFinder.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>

using namespace llvm;

Type* Types::voidType(Module* module)
{
    return Type::getVoidTy(module->getContext());
}

Type* Types::int8(Module* module)
{
    return Type::getInt8Ty(module->getContext());
}
llvm::Type *Types::int32(Module* module)
{
    return Type::getInt32Ty(module->getContext());
}
Type* Types::int64(Module* module)
{
    return Type::getInt64Ty(module->getContext());
}
Type* Types::boolType(Module* module)
{
    return Type::getInt1Ty(module->getContext());
}

PointerType* Types::voidPtr(Module* module)
{
    return Types::int8Ptr(module);
}
PointerType* Types::int8Ptr(Module* module)
{
    return Types::int8(module)->getPointerTo();
}
PointerType* Types::int32Ptr(Module* module)
{
    return Types::int32(module)->getPointerTo();
}
PointerType* Types::int64Ptr(Module* module)
{
    return Types::int64(module)->getPointerTo();
}

llvm::StructType* Types::getStruct(llvm::Module* module, const std::string& name)
{
    TypeFinder structFinder;
    structFinder.run(*module, true);
    for (auto* structType : structFinder)
    {
        if (structType->getStructName() == "struct." + name)
        {
            return structType;
        }
    }

    return nullptr;
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
