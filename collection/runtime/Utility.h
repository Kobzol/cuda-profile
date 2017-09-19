#pragma once

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdint>
#include <chrono>

namespace cupr
{
    inline void checkCudaCall(cudaError_t code, const char* file, int line)
    {
        if (code != cudaSuccess)
        {
            fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        }
    }
    inline int64_t getTimestamp()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
}

#define CHECK_CUDA_CALL(ans) { cupr::checkCudaCall((ans), __FILE__, __LINE__); }
#define __universal__ __device__ __host__

// CUDA device declarations for intellisense
__device__ unsigned int atomicInc(unsigned int* address, unsigned int val);
__device__ long long clock64();
