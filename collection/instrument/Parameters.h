#pragma once

#include <cstdint>

namespace cupr
{
    class Parameters
    {
    public:
        static bool shouldInstrumentLocals();

    private:
        static bool isParameterEnabled(const char* name);
    };
}
