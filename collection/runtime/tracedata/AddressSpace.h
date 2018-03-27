#pragma once

#include <cstdint>

namespace cupr
{
    enum class AddressSpace : uint8_t
    {
        Global = 0,
        Shared = 1,
        Constant = 2
    };
}
