#include "Parameters.h"

#include <cstdlib>
#include <cstring>
#include <string>

using namespace cupr;

static uint32_t BUFFER_SIZE_DEFAULT = 1024 * 1024;

uint32_t Parameters::getBufferSize()
{
    char* envBufferSize = getenv("CUPR_BUFFER_SIZE");
    if (envBufferSize == nullptr) return BUFFER_SIZE_DEFAULT;
    return static_cast<uint32_t>(std::stoi(envBufferSize));
}

bool Parameters::isPrettifyEnabled()
{
    return isParameterEnabled("CUPR_PRETTIFY");
}

bool Parameters::isProtobufEnabled()
{
    return isParameterEnabled("CUPR_PROTOBUF");
}

bool Parameters::isMappedMemoryEnabled()
{
    return isParameterEnabled("CUPR_MAPPED_MEMORY");
}

bool Parameters::isParameterEnabled(const char* name)
{
    char* parameter = getenv(name);
    return parameter != nullptr && strlen(parameter) > 0 && parameter[0] == '1';
}
