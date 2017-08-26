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
static std::vector<AllocRecord> allocations;

static __device__ AccessRecord* deviceAccessRecords;
static __device__ uint32_t deviceAccessRecordIndex;
static __device__ AllocRecord* deviceSharedBuffers;
static __device__ uint32_t deviceSharedBufferIndex;

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
extern "C" {
    void PREFIX(initKernelContext)(KernelContext* context, const char* kernelName)
    {
        context->kernelName = kernelName;
        context->timer = new CudaTimer();
    }
    void PREFIX(disposeKernelContext)(KernelContext* context)
    {
        delete context->timer;
    }
    void PREFIX(kernelStart)(KernelContext* context)
    {
        cudaMalloc((void**) &context->deviceAccessRecords, sizeof(AccessRecord) * BUFFER_SIZE);
        cudaMalloc((void**) &context->deviceSharedBuffers, sizeof(AllocRecord) * BUFFER_SIZE);

        const uint32_t zero = 0;
        CHECK_CUDA_CALL(cudaMemcpyToSymbol(deviceAccessRecords, &context->deviceAccessRecords, sizeof(context->deviceAccessRecords)));
        CHECK_CUDA_CALL(cudaMemcpyToSymbol(deviceAccessRecordIndex, &zero, sizeof(zero)));
        CHECK_CUDA_CALL(cudaMemcpyToSymbol(deviceSharedBuffers, &context->deviceSharedBuffers, sizeof(context->deviceSharedBuffers)));
        CHECK_CUDA_CALL(cudaMemcpyToSymbol(deviceSharedBufferIndex, &zero, sizeof(zero)));

        context->timer->start();
    }
    void PREFIX(kernelEnd)(KernelContext* context)
    {
        context->timer->stop_wait();
        CHECK_CUDA_CALL(cudaDeviceSynchronize());

        uint32_t accessCount = 0, sharedBuffersCount = 0;;
        CHECK_CUDA_CALL(cudaMemcpyFromSymbol(&accessCount, deviceAccessRecordIndex, sizeof(uint32_t)));
        CHECK_CUDA_CALL(cudaMemcpyFromSymbol(&sharedBuffersCount, deviceSharedBufferIndex, sizeof(uint32_t)));

        std::vector<AccessRecord> records(accessCount);
        if (accessCount > 0)
        {
            CHECK_CUDA_CALL(cudaMemcpy(records.data(), context->deviceAccessRecords, sizeof(AccessRecord) * accessCount, cudaMemcpyDeviceToHost));
        }
        CHECK_CUDA_CALL(cudaFree(context->deviceAccessRecords));

        std::vector<AllocRecord> sharedBuffers(sharedBuffersCount);
        if (sharedBuffersCount > 0)
        {
            CHECK_CUDA_CALL(cudaMemcpy(sharedBuffers.data(), context->deviceSharedBuffers, sizeof(AllocRecord) * sharedBuffersCount, cudaMemcpyDeviceToHost));
        }
        CHECK_CUDA_CALL(cudaFree(context->deviceSharedBuffers));

        for (auto& alloc: allocations)
        {
            sharedBuffers.push_back(alloc);
        }

        emitKernelData(context->kernelName, records, sharedBuffers, context->timer->get_time());
    }
    void PREFIX(malloc)(void* address, size_t size, size_t elementSize, const char* type)
    {
        allocations.emplace_back(address, size, elementSize, AddressSpace::Global, type);
    }
    void PREFIX(free)(void* address)
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
    extern "C" __device__ void PREFIX(store)(void* address, size_t size, uint32_t addressSpace,
                                             const char* type, int32_t debugIndex)
    {
        uint32_t index = atomicInc(&deviceAccessRecordIndex, BUFFER_SIZE);
        deviceAccessRecords[index] = AccessRecord(AccessType::Write, blockIdx, threadIdx, warpid(),
                                            address, size, static_cast<AddressSpace>(addressSpace),
                                            static_cast<int64_t>(clock64()), type, debugIndex);
    }
    extern "C" __device__ void PREFIX(load)(void* address, size_t size, uint32_t addressSpace,
                                            const char* type, int32_t debugIndex)
    {
        uint32_t index = atomicInc(&deviceAccessRecordIndex, BUFFER_SIZE);
        deviceAccessRecords[index] = AccessRecord(AccessType::Read, blockIdx, threadIdx, warpid(),
                                            address, size, static_cast<AddressSpace>(addressSpace),
                                            static_cast<int64_t>(clock64()), type, debugIndex);
    }
    extern "C" __device__ bool PREFIX(isFirstThread)()
    {
        return threadIdx.x == 0 &&
                threadIdx.y == 0 &&
                threadIdx.z == 0 &&
                blockIdx.x == 0 &&
                blockIdx.y == 0 &&
                blockIdx.z == 0;
    }
    extern "C" __device__ void PREFIX(markSharedBuffer)(void* address, size_t size, size_t elementSize,
                                                        const char* type)
    {
        printf("size: %lu, elementSize: %lu\n", size, elementSize);
        uint32_t index = atomicInc(&deviceSharedBufferIndex, BUFFER_SIZE);
        deviceSharedBuffers[index] = AllocRecord(address, size, elementSize, AddressSpace::Shared, type);
    }
}
