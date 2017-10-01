#pragma once

#include <cstddef>
#include "AddressSpace.h"
#include "../Utility.h"

namespace cupr
{
    struct AllocRecord
    {
    public:
        AllocRecord() = default;

        AllocRecord(void* address, size_t size, size_t elementSize,
                    AddressSpace addressSpace, const char* type)
                : address(address), size(size), elementSize(elementSize),
                  addressSpace(addressSpace), type(type)
        {

        }

        __device__ AllocRecord(void* address, size_t size, size_t elementSize,
                               AddressSpace addressSpace, size_t type)
                : address(address), size(size), elementSize(elementSize),
                  addressSpace(addressSpace), typeIndex(type)
        {

        }

        void* address = nullptr;
        size_t size = 0;
        size_t elementSize = 0;
        const char* type = nullptr;
        size_t typeIndex = 0;
        bool active = true;
        AddressSpace addressSpace = AddressSpace::Global;
    };
}
