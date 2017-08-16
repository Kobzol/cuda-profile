#pragma once

#include <cstddef>
#include <vector>
#include <iostream>

#include "prefix.h"
#include "AccessRecord.h"
#include "AllocRecord.h"
#include "format.h"
#include "../general.h"


static AccessRecord* deviceRecords = nullptr;
static size_t bufferSize = 1024;
static size_t kernelCounter = 0;
static std::vector<AllocRecord> allocations;

__device__ AccessRecord* devRecordsPtr;
__device__ uint32_t devRecordIndex;

static void emitKernelData(const std::string& kernelName,
                           const std::vector<AccessRecord>& records,
                           const std::vector<AllocRecord>& allocations)
{
    std::cerr << "Emmitted " << records.size() << " accesses " << "in kernel " << kernelName << std::endl;

    std::fstream kernelOutput(std::string(kernelName) + "-" + std::to_string(kernelCounter++) + ".json", std::fstream::out);
    outputKernelRun(kernelOutput, records, allocations);
    kernelOutput.flush();
}

extern "C" void PREFIX(kernelStart)()
{
    cudaMalloc((void**) &deviceRecords, sizeof(AccessRecord) * bufferSize);

    const uint32_t zero = 0;
    CHECK_CUDA_CALL(cudaMemcpyToSymbol(devRecordsPtr, &deviceRecords, sizeof(deviceRecords)));
    CHECK_CUDA_CALL(cudaMemcpyToSymbol(devRecordIndex, &zero, sizeof(zero)));
}
extern "C" void PREFIX(kernelEnd)(const char* kernelName)
{
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    uint32_t count = 0;
    CHECK_CUDA_CALL(cudaMemcpyFromSymbol(&count, devRecordIndex, sizeof(uint32_t)));

    std::vector<AccessRecord> records(count);
    CHECK_CUDA_CALL(cudaMemcpy(records.data(), deviceRecords, sizeof(AccessRecord) * count, cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaFree(deviceRecords));

    emitKernelData(kernelName, records, allocations);
}
extern "C" void PREFIX(malloc)(void* address, size_t size, size_t elementSize, const char* type)
{
    allocations.emplace_back(address, size, elementSize, type);
}
extern "C" void PREFIX(free)(void* address)
{
    for (auto& alloc: allocations)
    {
        if (alloc.address == address)
        {
            alloc.active = false;
        }
    }
}

__forceinline__ __device__ unsigned warpid()
{
    unsigned ret;
    asm volatile ("mov.u32 %0, %%warpid;" : "=r"(ret));
    return ret;
}
extern "C" __device__ void PREFIX(store)(void* address, size_t size, const char* type)
{
    uint32_t index = atomicInc(&devRecordIndex, 1024);
    devRecordsPtr[index] = AccessRecord(AccessType::Write, blockIdx, threadIdx, warpid(), address, size,
                                        static_cast<int64_t>(clock64()), type);
}
extern "C" __device__ void PREFIX(load)(void* address, size_t size, const char* type)
{
    uint32_t index = atomicInc(&devRecordIndex, 1024);
    devRecordsPtr[index] = AccessRecord(AccessType::Read, blockIdx, threadIdx, warpid(), address, size,
                                        static_cast<int64_t>(clock64()), type);
}
