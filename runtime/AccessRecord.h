#pragma once

#include "cudautil.h"
#include <cstdint>
#include <ostream>

enum class AccessType
{
    Read = 0,
    Write = 1
};

struct AccessRecord
{
public:
    AccessRecord() = default;
    __universal__ AccessRecord(AccessType accessType, dim3 blockIdx, dim3 threadIdx,
                               uint32_t warpId, void* memory, size_t size,
                               int64_t timestamp):
            accessType(accessType), blockIdx(blockIdx), threadIdx(threadIdx),
            warpId(warpId), address(memory), size(size), timestamp(timestamp)
    {

    }

    dim3 blockIdx;
    dim3 threadIdx;

    AccessType accessType = AccessType::Read;
    void* address = nullptr;
    size_t size = 0;
    uint32_t warpId = 0;

    int64_t timestamp = 0;
};
