#pragma once

#include <string>
#include <regex>

namespace llvm {
    class Function;
}

class RegexFilter
{
public:
    explicit RegexFilter(const std::string& regex);

    bool matchesFunction(llvm::Function* function);
    bool matches(const std::string& input) const;

private:
    std::string regexInput;
    std::regex regex;
};
