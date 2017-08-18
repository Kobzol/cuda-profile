#pragma once

#include <cstddef>

struct AllocRecord
{
public:
    AllocRecord() = default;
    AllocRecord(void* address, size_t size, size_t elementSize, const char* type)
            : address(address), size(size), elementSize(elementSize), type(type)
    {

    }

    void* address = nullptr;
    size_t size = 0;
    size_t elementSize = 0;
    const char* type = nullptr;
    bool active = true;
};
