#include "Parameters.h"

#include <cstdlib>
#include <cstring>
#include <string>

using namespace cupr;

const char* Parameters::PROTOBUF_FORMAT = "PROTOBUF";
const char* Parameters::CAPNP_FORMAT = "CAPNP";
const char* Parameters::JSON_FORMAT = "JSON";

static uint32_t BUFFER_SIZE_DEFAULT = 1024 * 1024;

uint32_t Parameters::getBufferSize()
{
    char* envBufferSize = getenv("BUFFER_SIZE");
    if (envBufferSize == nullptr) return BUFFER_SIZE_DEFAULT;
    return static_cast<uint32_t>(std::stoi(envBufferSize));
}

bool Parameters::isPrettifyEnabled()
{
    return isParameterEnabled("PRETTIFY");
}

bool Parameters::isCompressionEnabled()
{
#ifdef CUPR_USE_ZLIB
    return isParameterEnabled("COMPRESS");
#else
    return false;
#endif
}

std::string Parameters::getFormat()
{
    std::string format = Parameters::JSON_FORMAT;
    auto input = Parameters::getParameterString("FORMAT");
#ifdef CUPR_USE_CAPNP
    if (input && *input == Parameters::CAPNP_FORMAT)
    {
        format = Parameters::CAPNP_FORMAT;
    }
#endif
#ifdef CUPR_USE_PROTOBUF
    if (input && *input == Parameters::PROTOBUF_FORMAT)
    {
        format = Parameters::PROTOBUF_FORMAT;
    }
#endif

    return format;
}

bool Parameters::isMappedMemoryEnabled()
{
    return isParameterEnabled("HOST_MEMORY");
}

bool Parameters::isOutputEnabled()
{
    return !Parameters::isParameterEnabled("DISABLE_OUTPUT");
}

bool Parameters::isParameterEnabled(const char* name)
{
    char* parameter = getenv(name);
    return  parameter != nullptr &&
            strlen(parameter) > 0 &&
            parameter[0] == '1';
}

std::unique_ptr<std::string> Parameters::getParameterString(const char* name)
{
    char* parameter = getenv(name);
    if (parameter == nullptr) return std::unique_ptr<std::string>();

    return std::make_unique<std::string>(parameter);
}
