#pragma once

#include <cstddef>
#include <vector>
#include <iostream>

#include "prefix.h"
#include "AccessRecord.h"
#include "format.h"


static AccessRecord* deviceRecords = nullptr;
static size_t bufferSize = 1024;
static size_t kernelCounter = 0;
__device__ AccessRecord* devRecordsPtr;
__device__ uint32_t devRecordIndex;

static void emitKernelData(const std::string& kernelName, const std::vector<AccessRecord>& records)
{
    std::fstream kernelOutput(std::string(kernelName) + "-" + std::to_string(kernelCounter++) + ".json", std::fstream::out);
    kernelOutput << records << std::endl;
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
    std::vector<AccessRecord> records;
    records.resize(bufferSize);
    uint32_t count = 0;

    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    CHECK_CUDA_CALL(cudaMemcpy(records.data(), deviceRecords, sizeof(AccessRecord) * bufferSize, cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaMemcpyFromSymbol(&count, devRecordIndex, sizeof(uint32_t)));
    CHECK_CUDA_CALL(cudaFree(deviceRecords));
    deviceRecords = nullptr;

    emitKernelData(kernelName, records);
}

extern "C" __device__ void PREFIX(store)(uint32_t blockX, uint32_t blockY, uint32_t blockZ,
                                         uint32_t threadX, uint32_t threadY, uint32_t threadZ,
                                         void *address, size_t size)
{
    uint32_t index = atomicInc(&devRecordIndex, 1024);
    devRecordsPtr[index] = AccessRecord(AccessType::Write, dim3(blockX, blockY, blockZ), dim3(threadX, threadY, threadZ),
                                       address, size, static_cast<int64_t>(clock64()));
}
extern "C" __device__ void PREFIX(load)(uint32_t blockX, uint32_t blockY, uint32_t blockZ,
                                         uint32_t threadX, uint32_t threadY, uint32_t threadZ,
                                         void *address, size_t size)
{
    uint32_t index = atomicInc(&devRecordIndex, 1024);
    devRecordsPtr[index] = AccessRecord(AccessType::Read, dim3(blockX, blockY, blockZ), dim3(threadX, threadY, threadZ),
                                       address, size, static_cast<int64_t>(clock64()));
}
