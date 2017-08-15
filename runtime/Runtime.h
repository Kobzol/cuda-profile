#pragma once

#include <cstddef>
#include <vector>
#include <iostream>

#include "prefix.h"
#include "StoreRecord.h"
#include "format.h"


static StoreRecord* deviceRecords = nullptr;
static size_t bufferSize = 1024;
static size_t kernelCounter = 0;
__device__ StoreRecord* devRecordsPtr;
__device__ uint32_t devRecordIndex;

extern "C" void PREFIX(kernelStart)()
{
    cudaMalloc((void**) &deviceRecords, sizeof(StoreRecord) * bufferSize);

    const uint32_t zero = 0;
    cudaMemcpyToSymbol(devRecordsPtr, &deviceRecords, sizeof(deviceRecords));
    cudaMemcpyToSymbol(devRecordIndex, &zero, sizeof(zero));
}
extern "C" void PREFIX(kernelEnd)()
{
    std::vector<StoreRecord> records;
    records.resize(bufferSize);
    uint32_t count = 0;

    cudaDeviceSynchronize();
    CHECK_CUDA_CALL(cudaMemcpy(records.data(), deviceRecords, sizeof(StoreRecord) * bufferSize, cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaMemcpyFromSymbol(&count, devRecordIndex, sizeof(uint32_t)));

    for (int i = 0; i < count; i++)
    {
        std::cout << records[i] << std::endl;
    }

    cudaFree(deviceRecords);
    deviceRecords = nullptr;

    std::fstream kernelOutput("kernel-" + std::to_string(kernelCounter++) + ".json", std::fstream::out);
    kernelOutput << records << std::endl;
    kernelOutput.flush();
}

extern "C" __device__ void PREFIX(store)(uint32_t blockX, uint32_t blockY, uint32_t blockZ,
                                         uint32_t threadX, uint32_t threadY, uint32_t threadZ,
                                         void *address, size_t size)
{
    uint32_t index = atomicInc(&devRecordIndex, 1024);
    devRecordsPtr[index] = StoreRecord(AccessType::Write, dim3(blockX, blockY, blockZ), dim3(threadX, threadY, threadZ),
                                       address, size, static_cast<int64_t>(clock64()));
}
