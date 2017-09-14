#include "Demangler.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <cxxabi.h>

llvm::Function* Demangler::getFunctionByDemangledName(llvm::Module* module, std::string name) const
{
    for (auto& fn : module->getFunctionList())
    {
        std::string demangled = this->demangle(fn.getName().str());

        if (demangled.empty()) continue;

        demangled = demangled.substr(0, demangled.find('('));

        if (demangled == name)
        {
            return &fn;
        }
    }

    return nullptr;
}

std::string Demangler::demangle(std::string name) const
{
    char* demangled = abi::__cxa_demangle(name.c_str(), nullptr, nullptr, nullptr);

    std::string result = name;
    if (demangled != nullptr)
    {
        result = std::string(demangled);
    }
    else result = name + "()";

    ::free(demangled);
    return result;
}
