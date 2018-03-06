#include "Parameters.h"

#include <cstring>

bool Parameters::shouldInstrumentLocals()
{
    return Parameters::isParameterEnabled("CUPR_INSTRUMENT_LOCALS");
}
std::string Parameters::kernelRegex()
{
    auto regex = getenv("CUPR_KERNEL_REGEX");
    if (regex == nullptr) return "";
    return std::string(regex);
}

bool Parameters::isParameterEnabled(const char* name)
{
    char* parameter = getenv(name);
    return  parameter != nullptr &&
            strlen(parameter) > 0 &&
            parameter[0] == '1';
}
