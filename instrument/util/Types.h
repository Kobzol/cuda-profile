#pragma once

#include <string>

namespace llvm {
    class Module;
    class PointerType;
    class StructType;
    class Type;
}

class Types
{
public:
    static llvm::Type* voidType(llvm::Module* module);

    static llvm::Type* int8(llvm::Module* module);
    static llvm::Type* int32(llvm::Module* module);
    static llvm::Type* int64(llvm::Module* module);
    static llvm::Type* boolType(llvm::Module* module);

    static llvm::PointerType* voidPtr(llvm::Module* module);
    static llvm::PointerType* int8Ptr(llvm::Module* module);
    static llvm::PointerType* int32Ptr(llvm::Module* module);
    static llvm::PointerType* int64Ptr(llvm::Module* module);

    static llvm::StructType* getStruct(llvm::Module* module, const std::string& name);

    static std::string print(llvm::Type* type);
};
