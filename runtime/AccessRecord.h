#pragma once

#include <cstdint>

#include "cudautil.h"
#include "AddressSpace.h"

enum class AccessType: uint32_t
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
                               AddressSpace addressSpace, int64_t timestamp, const char* type,
                               int32_t debugIndex):
            accessType(accessType), blockIdx(blockIdx), threadIdx(threadIdx),
            warpId(warpId), address(memory), size(size),
            addressSpace(addressSpace), timestamp(timestamp), debugIndex(debugIndex)
    {
        int i = 0;
        for (; i < sizeof(this->type) - 1; i++)
        {
            if (*type == '\0') break;
            this->type[i] = *type++;
        }
        this->type[i] = '\0';
    }

    uint3 blockIdx { 0, 0, 0 };
    uint3 threadIdx { 0, 0, 0 };

    void* address = nullptr;
    size_t size = 0;
    int64_t timestamp = 0;
    uint32_t warpId = 0;
    AccessType accessType = AccessType::Read;
    int32_t debugIndex = 0;
    AddressSpace addressSpace = AddressSpace::Global;

    char type[32] = {};
};
