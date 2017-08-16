#pragma once

#include <cstddef>

struct AllocRecord
{
public:
    AllocRecord() = default;
    AllocRecord(void* address, size_t size, const char* type)
            : address(address), size(size), type(type)
    {

    }

    void* address;
    size_t size;
    const char* type;
    bool active = true;
};
