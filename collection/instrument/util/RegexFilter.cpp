#include "RegexFilter.h"
#include "Demangler.h"

#include <llvm/IR/Function.h>

using namespace llvm;

RegexFilter::RegexFilter(const std::string& regex): regexInput(regex), regex(regexInput)
{

}

bool RegexFilter::matches(const std::string& input) const
{
    if (this->regexInput.empty()) return true;

    return std::regex_match(input, this->regex);
}

bool RegexFilter::matchesFunction(Function* function)
{
    Demangler demangler;
    auto name = demangler.demangle(function->getName().str());
    return this->matches(name);
}
