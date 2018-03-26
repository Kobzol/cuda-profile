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

        AllocRecord(void* address, size_t size, AddressSpace addressSpace)
                : address(address), size(size), addressSpace(addressSpace)
        {

        }
        AllocRecord(void* address, size_t size, size_t elementSize,
                    AddressSpace addressSpace, const char* type,
                    const char* name, const char* location)
                : address(address), size(size), elementSize(elementSize),
                  addressSpace(addressSpace), type(type),
                  name(name), location(location)
        {

        }

        __device__ AllocRecord(void* address, size_t size, size_t elementSize,
                               AddressSpace addressSpace, size_t type, size_t nameIndex)
                : address(address), size(size), elementSize(elementSize), addressSpace(addressSpace),
                  typeIndex(static_cast<int32_t>(type)), nameIndex(static_cast<int32_t>(nameIndex))
        {

        }

        void* address = nullptr;
        size_t size = 0;
        size_t elementSize = 0;
        const char* type = nullptr;
        const char* name = nullptr;
        const char* location = nullptr;
        int32_t typeIndex = -1;
        bool active = true;
        AddressSpace addressSpace = AddressSpace::Global;
        int32_t nameIndex = -1;
    };
}
