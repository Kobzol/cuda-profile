#include "TraceFormatter.h"

static const char* digits = "0123456789ABCDEF";

//https://stackoverflow.com/a/33447587/1107768
std::string cupr::TraceFormatter::hexPointer(const void* ptr)
{
    auto w = reinterpret_cast<size_t>(ptr);
    std::string hexString(18, '0');
    hexString[1] = 'x';

    for (size_t i = 0, j = 15 * 4; i < 16; ++i, j -= 4)
    {
        hexString[i + 2] = digits[(w >> j) & 0x0F];
    }

    return hexString;
}
