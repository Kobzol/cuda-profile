#pragma once

#include <cuda_runtime_api.h>
#include <cstdio>

#include "cudautil.h"

static int* devMemory;
static __device__ int* devPointer;

extern "C" void PREFIX(initMemory)()
{
    cudaMalloc((void**) &devMemory, sizeof(int));
    cudaMemcpyToSymbol(devPointer, &devMemory, sizeof(devMemory));
}

extern "C" __device__ void PREFIX(store)(int threadId, void* address)
{
    *devPointer = threadId;
}

extern "C" void PREFIX(kernelStart)()
{

}
extern "C" void PREFIX(kernelEnd)()
{
    int data;

    cudaDeviceSynchronize();
    CHECK_CUDA(cudaMemcpy(&data, devMemory, sizeof(int), cudaMemcpyDeviceToHost));
    printf("On host: %d\n", data);
}
