#include "TraceFormatter.h"

std::string cupr::TraceFormatter::hexPointer(const void* ptr)
{
    std::ostringstream address;
    address << ptr;
    return address.str();
}
