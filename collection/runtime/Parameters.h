#pragma once

#include <cstdint>
#include <string>
#include <memory>

namespace cupr
{
    class Parameters
    {
    public:
        static const char* PROTOBUF_FORMAT;
        static const char* CAPNP_FORMAT;
        static const char* JSON_FORMAT;

        static uint32_t getBufferSize();
        static bool isPrettifyEnabled();
        static bool isCompressionEnabled();
        static std::string getFormat();
        static bool isMappedMemoryEnabled();
        static bool isOutputEnabled();

    private:
        static bool isParameterEnabled(const char* name);
        static std::unique_ptr<std::string> getParameterString(const char* name);
    };
}
