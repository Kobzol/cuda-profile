#include "Parameters.h"

#include <cstdlib>
#include <cstring>

using namespace cupr;

bool Parameters::shouldInstrumentLocals()
{
    return Parameters::isParameterEnabled("CUPR_INSTRUMENT_LOCALS");
}

bool Parameters::isParameterEnabled(const char* name)
{
    char* parameter = getenv(name);
    return  parameter != nullptr &&
            strlen(parameter) > 0 &&
            parameter[0] == '1';
}
