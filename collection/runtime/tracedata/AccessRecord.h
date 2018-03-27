#pragma once

#include <cstdint>

#include "../Utility.h"
#include "AddressSpace.h"

namespace cupr
{
    enum class AccessType : uint8_t
    {
        Read = 0,
        Write = 1
    };

    struct AccessRecord
    {
    public:
        AccessRecord() = default;

        __universal__ AccessRecord(AccessType accessType, uint3 blockIdx, uint3 threadIdx,
                                   uint32_t warpId, void* address, uint8_t size,
                                   AddressSpace addressSpace, int64_t timestamp, uint8_t type,
                                   int32_t debugIndex, uint64_t value) :
                blockIdx(blockIdx), threadIdx(threadIdx), address(address), timestamp(timestamp),
                warpId(warpId), debugIndex(debugIndex), type(type), value(value),
                addressSpace(addressSpace), kind(accessType), size(size)
        {

        }

        uint3 blockIdx;
        uint3 threadIdx;

        void* address;
        int64_t timestamp;
        uint32_t warpId;
        int32_t debugIndex;

        size_t type;
        uint64_t value;

        AddressSpace addressSpace;
        AccessType kind;
        uint8_t size;
    };
}
