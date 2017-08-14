#pragma once

#include "cudautil.h"
#include "StoreRecord.h"
#include <cstddef>
#include <vector>
#include <iostream>

static StoreRecord* devMemory;
static size_t bufferSize = 1024;
static __device__ StoreRecord* devPointer;
static __device__ uint32_t storeIndex = 0;

extern "C" void PREFIX(initMemory)()
{
    cudaMalloc((void**) &devMemory, sizeof(StoreRecord) * bufferSize);
    cudaMemcpyToSymbol(devPointer, &devMemory, sizeof(devMemory));
}

extern "C" void PREFIX(kernelStart)()
{

}

extern "C" void PREFIX(kernelEnd)()
{
    StoreRecord* data = new StoreRecord[1024];

    cudaDeviceSynchronize();
    CHECK_CUDA(cudaMemcpy(data, devMemory, sizeof(StoreRecord) * bufferSize, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 20; i++)
    {
        std::cout << data[i] << std::endl;
    }
}

extern "C" __device__ void PREFIX(store)(uint32_t blockX, uint32_t blockY, uint32_t blockZ,
                                         uint32_t threadX, uint32_t threadY, uint32_t threadZ,
                                         void *address, size_t size)
{
    uint32_t index = atomicInc(&storeIndex, 1024);
    devPointer[index] = StoreRecord(dim3(blockX, blockY, blockZ), dim3(threadX, threadY, threadZ), address, size);
}
