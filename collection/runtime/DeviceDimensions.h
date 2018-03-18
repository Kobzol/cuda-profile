#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

struct DeviceDimensions
{
    dim3 grid;
    dim3 block;
    uint32_t warpSize;
    uint32_t bankSize;
};
