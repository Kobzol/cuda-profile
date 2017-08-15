#pragma once

#include <string>

namespace llvm {
    class Function;
    class Module;
}

class Demangler
{
public:
    llvm::Function* getFunctionByDemangledName(llvm::Module* module, std::string name) const;
    std::string demangle(std::string name) const;
};
