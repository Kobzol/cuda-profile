#pragma once

#include <string>
#include <unordered_map>

namespace llvm {
    class GlobalVariable;
    class Module;
    class PointerType;
    class StructType;
    class Type;
}

class Types
{
public:
    Types() = default;
    explicit Types(llvm::Module* module): module(module)
    {

    }

    void setModule(llvm::Module* module)
    {
        this->module = module;
        this->structMap.clear();
    }
    
    llvm::Type* voidType();

    llvm::Type* int8();
    llvm::Type* int32();
    llvm::Type* int64();
    llvm::Type* boolType();

    llvm::PointerType* voidPtr();
    llvm::PointerType* int8Ptr();
    llvm::PointerType* int32Ptr();
    llvm::PointerType* int64Ptr();

    llvm::StructType* getCompositeType(const std::string& name);
    void getGlobalVariableSize(llvm::GlobalVariable* globalVariable, size_t& size, size_t& elementSize);

    std::string stringify(const llvm::Type* type);
    
private:
    llvm::Module* module = nullptr;

    std::unordered_map<std::string, llvm::StructType*> structMap;
};
