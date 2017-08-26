#pragma once

#include <cstdint>

enum class AddressSpace: uint32_t
{
    Global = 0,
    Shared = 1,
    Constant = 2
};
