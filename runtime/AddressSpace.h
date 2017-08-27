#pragma once

#include <cstdint>

namespace cupr
{
    enum class AddressSpace : uint32_t
    {
        Global = 0,
        Shared = 1,
        Constant = 2
    };
}
