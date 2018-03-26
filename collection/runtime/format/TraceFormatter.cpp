#include "TraceFormatter.h"

//https://stackoverflow.com/a/33447587/1107768
template <typename I>
std::string n2hexstr(I w, size_t hex_len = sizeof(I)<<1)
{
    static const char* digits = "0123456789ABCDEF";
    std::string rc(hex_len, '0');

    for (size_t i = 0, j = (hex_len - 1) * 4; i < hex_len; ++i, j -= 4)
    {
        rc[i] = digits[(w >> j) & 0x0F];
    }

    return "0x" + rc;
}

std::string cupr::TraceFormatter::hexPointer(const void* ptr)
{
    return n2hexstr(reinterpret_cast<size_t>(ptr));
}
