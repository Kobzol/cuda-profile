#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../runtime/Prefix.h"
#include "../runtime/tracedata/AccessRecord.h"
#include "../runtime/tracedata/AllocRecord.h"
#include "../runtime/format/TraceFormatter.h"
#include "../runtime/CudaTimer.h"
#include "../runtime/KernelContext.h"
#include "../runtime/tracedata/AddressSpace.h"
#include "../runtime/Utility.h"
#include "../runtime/RuntimeState.h"
#include "../runtime/Parameters.h"
#include "../runtime/DeviceDimensions.h"


#define ATOMIC_INSERT(buffer, index, maxSize, item) \
    do { \
        uint32_t oldIndex = atomicInc(index, maxSize); \
        if (*(index) < oldIndex) printf("DEVICE PROFILING OVERFLOW\n"); \
        (buffer)[oldIndex] = item; \
    } while (false)

namespace cupr
{
    static __device__ uint32_t deviceBufferSize;
    static __device__ cupr::AccessRecord* deviceAccessRecords;
    static __device__ uint32_t deviceAccessRecordIndex;
    static __device__ cupr::AllocRecord* deviceSharedBuffers;
    static __device__ uint32_t deviceSharedBufferIndex;

    /**
     * Grid dimension, block dimension, warpSize.
     */
    static __device__ DeviceDimensions* deviceDimensions;
}
extern "C"
{
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
        uint32_t bufferSize = cupr::Parameters::getBufferSize();

        typedef cudaError (*cudaAllocateType)(void**, size_t);

        cudaAllocateType allocFn = cupr::Parameters::isMappedMemoryEnabled() ? (cudaAllocateType) cudaMallocHost : (cudaAllocateType) cudaMalloc;
        CHECK_CUDA_CALL(allocFn((void**) &context->deviceAccessRecords, sizeof(cupr::AccessRecord) * bufferSize));
        CHECK_CUDA_CALL(allocFn((void**) &context->deviceSharedBuffers, sizeof(cupr::AllocRecord) * bufferSize));
        CHECK_CUDA_CALL(allocFn((void**) &context->deviceDimensions, sizeof(DeviceDimensions)));

        const uint32_t zero = 0;
        CHECK_CUDA_CALL(cudaMemcpyToSymbol(cupr::deviceAccessRecords, &context->deviceAccessRecords, sizeof(context->deviceAccessRecords)));
        CHECK_CUDA_CALL(cudaMemcpyToSymbol(cupr::deviceAccessRecordIndex, &zero, sizeof(zero)));
        CHECK_CUDA_CALL(cudaMemcpyToSymbol(cupr::deviceSharedBuffers, &context->deviceSharedBuffers, sizeof(context->deviceSharedBuffers)));
        CHECK_CUDA_CALL(cudaMemcpyToSymbol(cupr::deviceSharedBufferIndex, &zero, sizeof(zero)));
        CHECK_CUDA_CALL(cudaMemcpyToSymbol(cupr::deviceBufferSize, &bufferSize, sizeof(bufferSize)));
        CHECK_CUDA_CALL(cudaMemcpyToSymbol(cupr::deviceDimensions, &context->deviceDimensions, sizeof(context->deviceDimensions)));

        context->timer->start();
    }
    void CU_PREFIX(kernelEnd)(cupr::KernelContext* context)
    {
        context->timer->stop_wait();
        CHECK_CUDA_CALL(cudaDeviceSynchronize());

        uint32_t accessCount = 0, sharedBuffersCount = 0;;
        CHECK_CUDA_CALL(cudaMemcpyFromSymbol(&accessCount, cupr::deviceAccessRecordIndex, sizeof(cupr::deviceAccessRecordIndex)));
        CHECK_CUDA_CALL(cudaMemcpyFromSymbol(&sharedBuffersCount, cupr::deviceSharedBufferIndex, sizeof(cupr::deviceSharedBufferIndex)));

        std::vector<cupr::AccessRecord> records(accessCount);
        if (accessCount > 0)
        {
            CHECK_CUDA_CALL(cudaMemcpy(records.data(), context->deviceAccessRecords, sizeof(cupr::AccessRecord) * accessCount, cudaMemcpyDeviceToHost));
        }

        std::vector<cupr::AllocRecord> sharedBuffers(sharedBuffersCount);
        if (sharedBuffersCount > 0)
        {
            CHECK_CUDA_CALL(cudaMemcpy(sharedBuffers.data(), context->deviceSharedBuffers, sizeof(cupr::AllocRecord) * sharedBuffersCount, cudaMemcpyDeviceToHost));
        }

        DeviceDimensions dimensions;
        CHECK_CUDA_CALL(cudaMemcpy(&dimensions, context->deviceDimensions, sizeof(DeviceDimensions), cudaMemcpyDeviceToHost));

        cudaSharedMemConfig sharedMemConfig;
        CHECK_CUDA_CALL(cudaDeviceGetSharedMemConfig(&sharedMemConfig));

        dimensions.bankSize = sharedMemConfig == cudaSharedMemBankSizeEightByte ? 8 : 4;

        typedef cudaError (*cudaFreeType)(void*);

        cudaFreeType freeFn = cupr::Parameters::isMappedMemoryEnabled() ? (cudaFreeType) cudaFreeHost : (cudaFreeType) cudaFree;
        CHECK_CUDA_CALL(freeFn(context->deviceAccessRecords));
        CHECK_CUDA_CALL(freeFn(context->deviceSharedBuffers));
        CHECK_CUDA_CALL(freeFn(context->deviceDimensions));

        for (auto& alloc: cupr::state.getAllocations())
        {
            sharedBuffers.push_back(alloc);
        }

        cupr::state.getEmitter().emitKernelTrace(context->kernelName, dimensions,
                                            records, sharedBuffers, context->timer->get_time());
    }

    static __forceinline__ __device__ uint32_t warpid()
    {
        uint32_t ret;
        asm volatile ("mov.u32 %0, %%warpid;" : "=r"(ret));
        return ret;
    }
    extern "C" __device__ void CU_PREFIX(store)(void* address, size_t size, uint32_t addressSpace,
                                                size_t type, int32_t debugIndex, uint64_t value)
    {
        ATOMIC_INSERT(cupr::deviceAccessRecords,
                      &cupr::deviceAccessRecordIndex,
                      cupr::deviceBufferSize,
                      cupr::AccessRecord(cupr::AccessType::Write, blockIdx, threadIdx, warpid(),
                                         address, size, static_cast<cupr::AddressSpace>(addressSpace),
                                         static_cast<int64_t>(clock64()), type, debugIndex, value));
    }
    extern "C" __device__ void CU_PREFIX(load)(void* address, size_t size, uint32_t addressSpace,
                                               size_t type, int32_t debugIndex, uint64_t value)
    {
        ATOMIC_INSERT(cupr::deviceAccessRecords,
                      &cupr::deviceAccessRecordIndex,
                      cupr::deviceBufferSize,
                      cupr::AccessRecord(cupr::AccessType::Read, blockIdx, threadIdx, warpid(),
                              address, size, static_cast<cupr::AddressSpace>(addressSpace),
                              static_cast<int64_t>(clock64()), type, debugIndex, value));
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
    extern "C" __device__ void CU_PREFIX(storeDimensions)()
    {
        cupr::deviceDimensions->grid = gridDim;
        cupr::deviceDimensions->block = blockDim;
        cupr::deviceDimensions->warpSize = warpSize;
    }
    extern "C" __device__ void CU_PREFIX(markSharedBuffer)(void* address, size_t size, size_t elementSize,
                                                           size_t type, size_t nameIndex)
    {
        ATOMIC_INSERT(cupr::deviceSharedBuffers,
                      &cupr::deviceSharedBufferIndex,
                      cupr::deviceBufferSize,
                      cupr::AllocRecord(address, size, elementSize, cupr::AddressSpace::Shared, type, nameIndex));
    }
}
