#pragma once

#include <cstdint>
#include <string>

class Parameters
{
public:
    static bool shouldInstrumentLocals();
    static std::string kernelRegex();

private:
    static bool isParameterEnabled(const char* name);
};
