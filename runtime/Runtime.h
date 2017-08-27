#pragma once

#include <cstddef>
#include <vector>
#include <iostream>

#include <device_launch_parameters.h>

#include "prefix.h"
#include "AccessRecord.h"
#include "AllocRecord.h"
#include "format.h"
#include "CudaTimer.h"
#include "KernelContext.h"
#include "AddressSpace.h"

#define BUFFER_SIZE 1024

static size_t kernelCounter = 0;
static std::vector<cupr::AllocRecord> allocations;

static __device__ cupr::AccessRecord* deviceAccessRecords;
static __device__ uint32_t deviceAccessRecordIndex;
static __device__ cupr::AllocRecord* deviceSharedBuffers;
static __device__ uint32_t deviceSharedBufferIndex;

namespace cupr {
    inline static void emitKernelData(const std::string& kernelName,
                                      const std::vector<AccessRecord>& records,
                                      const std::vector<AllocRecord>& allocations,
                                      float kernelTime)
    {
        std::cerr << "Emmitted " << records.size() << " accesses " << "in kernel " << kernelName << std::endl;

        std::fstream kernelOutput(std::string(kernelName) + "-" + std::to_string(kernelCounter++) + ".json", std::fstream::out);

        Formatter formatter;
        formatter.outputKernelRun(kernelOutput, records, allocations, kernelTime);
        kernelOutput.flush();
    }
}
extern "C" {
    void CU_PREFIX(initKernelContext)(cupr::KernelContext* context, const char* kernelName)
    {
        context->kernelName = kernelName;
        context->timer = new cupr::CudaTimer();
    }
    void CU_PREFIX(disposeKernelContext)(cupr::KernelContext* context)
    {
        delete context->timer;
    }
    void CU_PREFIX(kernelStart)(cupr::KernelContext* context)
    {
        cudaMalloc((void**) &context->deviceAccessRecords, sizeof(cupr::AccessRecord) * BUFFER_SIZE);
        cudaMalloc((void**) &context->deviceSharedBuffers, sizeof(cupr::AllocRecord) * BUFFER_SIZE);

        const uint32_t zero = 0;
        CHECK_CUDA_CALL(cudaMemcpyToSymbol(deviceAccessRecords, &context->deviceAccessRecords, sizeof(context->deviceAccessRecords)));
        CHECK_CUDA_CALL(cudaMemcpyToSymbol(deviceAccessRecordIndex, &zero, sizeof(zero)));
        CHECK_CUDA_CALL(cudaMemcpyToSymbol(deviceSharedBuffers, &context->deviceSharedBuffers, sizeof(context->deviceSharedBuffers)));
        CHECK_CUDA_CALL(cudaMemcpyToSymbol(deviceSharedBufferIndex, &zero, sizeof(zero)));

        context->timer->start();
    }
    void CU_PREFIX(kernelEnd)(cupr::KernelContext* context)
    {
        context->timer->stop_wait();
        CHECK_CUDA_CALL(cudaDeviceSynchronize());

        uint32_t accessCount = 0, sharedBuffersCount = 0;;
        CHECK_CUDA_CALL(cudaMemcpyFromSymbol(&accessCount, deviceAccessRecordIndex, sizeof(uint32_t)));
        CHECK_CUDA_CALL(cudaMemcpyFromSymbol(&sharedBuffersCount, deviceSharedBufferIndex, sizeof(uint32_t)));

        std::vector<cupr::AccessRecord> records(accessCount);
        if (accessCount > 0)
        {
            CHECK_CUDA_CALL(cudaMemcpy(records.data(), context->deviceAccessRecords, sizeof(cupr::AccessRecord) * accessCount, cudaMemcpyDeviceToHost));
        }
        CHECK_CUDA_CALL(cudaFree(context->deviceAccessRecords));

        std::vector<cupr::AllocRecord> sharedBuffers(sharedBuffersCount);
        if (sharedBuffersCount > 0)
        {
            CHECK_CUDA_CALL(cudaMemcpy(sharedBuffers.data(), context->deviceSharedBuffers, sizeof(cupr::AllocRecord) * sharedBuffersCount, cudaMemcpyDeviceToHost));
        }
        CHECK_CUDA_CALL(cudaFree(context->deviceSharedBuffers));

        for (auto& alloc: allocations)
        {
            sharedBuffers.push_back(alloc);
        }

        emitKernelData(context->kernelName, records, sharedBuffers, context->timer->get_time());
    }
    void CU_PREFIX(malloc)(void* address, size_t size, size_t elementSize, const char* type)
    {
        allocations.emplace_back(address, size, elementSize, cupr::AddressSpace::Global, type);
    }
    void CU_PREFIX(free)(void* address)
    {
        for (auto& alloc: allocations)
        {
            if (alloc.address == address)
            {
                alloc.active = false;
            }
        }
    }

    __forceinline__ __device__ uint32_t warpid()
    {
        uint32_t ret;
        asm volatile ("mov.u32 %0, %%warpid;" : "=r"(ret));
        return ret;
    }
    extern "C" __device__ void CU_PREFIX(store)(void* address, size_t size, uint32_t addressSpace,
                                             size_t type, int32_t debugIndex)
    {
        uint32_t index = atomicInc(&deviceAccessRecordIndex, BUFFER_SIZE);
        deviceAccessRecords[index] = cupr::AccessRecord(cupr::AccessType::Write, blockIdx, threadIdx, warpid(),
                                            address, size, static_cast<cupr::AddressSpace>(addressSpace),
                                            static_cast<int64_t>(clock64()), type, debugIndex);
    }
    extern "C" __device__ void CU_PREFIX(load)(void* address, size_t size, uint32_t addressSpace,
                                            size_t type, int32_t debugIndex)
    {
        uint32_t index = atomicInc(&deviceAccessRecordIndex, BUFFER_SIZE);
        deviceAccessRecords[index] = cupr::AccessRecord(cupr::AccessType::Read, blockIdx, threadIdx, warpid(),
                                            address, size, static_cast<cupr::AddressSpace>(addressSpace),
                                            static_cast<int64_t>(clock64()), type, debugIndex);
    }
    extern "C" __device__ bool CU_PREFIX(isFirstThread)()
    {
        return threadIdx.x == 0 &&
                threadIdx.y == 0 &&
                threadIdx.z == 0 &&
                blockIdx.x == 0 &&
                blockIdx.y == 0 &&
                blockIdx.z == 0;
    }
    extern "C" __device__ void CU_PREFIX(markSharedBuffer)(void* address, size_t size, size_t elementSize,
                                                        size_t type)
    {
        uint32_t index = atomicInc(&deviceSharedBufferIndex, BUFFER_SIZE);
        deviceSharedBuffers[index] = cupr::AllocRecord(address, size, elementSize, cupr::AddressSpace::Shared, type);
    }
}
