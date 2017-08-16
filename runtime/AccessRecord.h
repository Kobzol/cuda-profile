#pragma once

#include "cudautil.h"
#include <cstdint>
#include <ostream>
#include <cstring>

enum class AccessType
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
                               int64_t timestamp, const char* type):
            accessType(accessType), blockIdx(blockIdx), threadIdx(threadIdx),
            warpId(warpId), address(memory), size(size),
            timestamp(timestamp)
    {
        int i = 0;
        for (; i < sizeof(this->type) - 1; i++)
        {
            if (*type == '\0') break;
            this->type[i] = *type++;
        }
        this->type[i] = '\0';
    }

    uint3 blockIdx;
    uint3 threadIdx;

    AccessType accessType = AccessType::Read;
    void* address = nullptr;
    size_t size = 0;
    char type[32];
    uint32_t warpId = 0;

    int64_t timestamp = 0;
};
