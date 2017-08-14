#pragma once

#include "cudautil.h"
#include <cstdint>
#include <ostream>


struct StoreRecord
{
public:
    StoreRecord() = default;
    __universal__ StoreRecord(dim3 blockIdx, dim3 threadIdx, void* memory, size_t size):
        blockIdx(blockIdx), threadIdx(threadIdx), memory(memory), size(size)
    {

    }

    dim3 blockIdx;
    dim3 threadIdx;

    void* memory;
    size_t size;
};

std::ostream& operator<<(std::ostream& os, const dim3& dimension)
{
    os << dimension.x << ";" << dimension.y << ";" << dimension.z;
    return os;
}

std::ostream& operator<<(std::ostream& os, StoreRecord& record)
{
    os << "Store(";
    os << record.blockIdx << ", ";
    os << record.threadIdx << ", ";
    os << record.memory << ", " << record.size << ")";

    return os;
}
