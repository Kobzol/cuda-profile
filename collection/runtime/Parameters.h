#pragma once

#include <cstdint>

namespace cupr
{
    class Parameters
    {
    public:
        static uint32_t getBufferSize();
        static bool isPrettifyEnabled();
        static bool isCompressionEnabled();
        static bool isProtobufEnabled();
        static bool isMappedMemoryEnabled();
        static bool isOutputEnabled();

    private:
        static bool isParameterEnabled(const char* name);
    };
}
