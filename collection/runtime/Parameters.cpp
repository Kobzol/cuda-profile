#include "Parameters.h"

#include <cstdlib>
#include <cstring>
#include <string>

using namespace cupr;

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

bool Parameters::isProtobufEnabled()
{
#ifdef CUPR_USE_PROTOBUF
    return isParameterEnabled("PROTOBUF");
#else
    return false;
#endif
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
