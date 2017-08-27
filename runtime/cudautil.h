#pragma once

#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <cstdio>

namespace cupr
{
    inline void checkCudaCall(cudaError_t code, const char* file, int line)
    {
        if (code != cudaSuccess)
        {
            fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        }
    }
}

#define CHECK_CUDA_CALL(ans) { cupr::checkCudaCall((ans), __FILE__, __LINE__); }
#define __universal__ __device__ __host__
