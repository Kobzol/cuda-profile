#pragma once

#include <cstdint>

#include "cudautil.h"
#include "AddressSpace.h"

namespace cupr
{
    enum class AccessType : uint32_t
    {
        Read = 0,
        Write = 1
    };

    struct AccessRecord
    {
    public:
        AccessRecord() = default;

        __universal__ AccessRecord(AccessType accessType, uint3 blockIdx, uint3 threadIdx,
                                   uint32_t warpId, void* memory, size_t size,
                                   AddressSpace addressSpace, int64_t timestamp, size_t type,
                                   int32_t debugIndex) :
                accessType(accessType), blockIdx(blockIdx), threadIdx(threadIdx),
                warpId(warpId), address(memory), size(size),
                addressSpace(addressSpace), timestamp(timestamp), type(type),
                debugIndex(debugIndex)
        {

        }

        uint3 blockIdx{0, 0, 0};
        uint3 threadIdx{0, 0, 0};

        void* address = nullptr;
        size_t size = 0;
        int64_t timestamp = 0;
        uint32_t warpId = 0;
        AccessType accessType = AccessType::Read;
        int32_t debugIndex = 0;
        AddressSpace addressSpace = AddressSpace::Global;

        size_t type = 0;
    };
}
