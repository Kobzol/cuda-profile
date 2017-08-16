#pragma once

#include <cstddef>

struct AllocRecord
{
public:
    AllocRecord() = default;
    AllocRecord(void* address, size_t size): address(address), size(size)
    {

    }

    void* address;
    size_t size;
    bool active = true;
};
