#pragma once

#include <cstddef>
#include <cstdlib>
#include <vector>
#include <iostream>

#include "prefix.h"
#include "AccessRecord.h"
#include "AllocRecord.h"
#include "format.h"
#include "CudaTimer.h"
#include "KernelContext.h"
#include "AddressSpace.h"
#include "cudautil.h"


#define ATOMIC_INSERT(buffer, index, maxSize, item) \
    do { \
        uint32_t oldIndex = atomicInc(index, maxSize); \
        if (*(index) < oldIndex) printf("DEVICE PROFILING OVERFLOW\n"); \
        (buffer)[oldIndex] = item; \
    } while (false)

namespace cupr {
    static uint32_t BUFFER_SIZE_DEFAULT = 1024 * 1024;

    static size_t kernelCounter = 0;
    static std::vector<cupr::AllocRecord> allocations;

    static __device__ uint32_t deviceBufferSize;
    static __device__ cupr::AccessRecord* deviceAccessRecords;
    static __device__ uint32_t deviceAccessRecordIndex;
    static __device__ cupr::AllocRecord* deviceSharedBuffers;
    static __device__ uint32_t deviceSharedBufferIndex;

    inline static void emitKernelDataJson(const std::string& fileName,
                                          const std::vector<AccessRecord>& records,
                                          const std::vector<AllocRecord>& allocations,
                                          float kernelTime)
    {
        std::fstream kernelOutput(fileName + ".json", std::fstream::out);

        char* prettifyEnv = getenv("CUPROFILE_PRETTIFY");
        bool prettify = prettifyEnv != nullptr && strlen(prettifyEnv) > 0 && prettifyEnv[0] == '1';

        Formatter formatter;
        formatter.outputKernelRunJson(kernelOutput, records, allocations, kernelTime, prettify);
        kernelOutput.flush();
    }
#ifdef USE_PROTOBUF
    inline static void emitKernelDataProtobuf(const std::string& fileName,
                                              const std::vector<AccessRecord>& records,
                                              const std::vector<AllocRecord>& allocations,
                                              float kernelTime)
    {
        std::fstream kernelOutput(fileName + ".protobuf", std::fstream::out);

        Formatter formatter;
        formatter.outputKernelRunProtobuf(kernelOutput, records, allocations, kernelTime);
        kernelOutput.flush();
    }
#endif
    inline static void emitKernelData(const std::string& kernelName,
                                      const std::vector<AccessRecord>& records,
                                      const std::vector<AllocRecord>& allocations,
                                      float kernelTime)
    {
        std::cerr << "Emmitted " << records.size() << " accesses " << "in kernel " << kernelName << std::endl;

        std::string kernelFile = std::string(kernelName) + "-" + std::to_string(kernelCounter++);
#ifdef USE_PROTOBUF
        char* outputProtobuf = getenv("CUPROFILE_PROTOBUF");
        bool protobuf = outputProtobuf != nullptr && strlen(outputProtobuf) > 0 && outputProtobuf[0] == '1';

        if (protobuf)
        {
            emitKernelDataProtobuf(kernelFile, records, allocations, kernelTime);
        }
        else
#endif
        {
            emitKernelDataJson(kernelFile, records, allocations, kernelTime);
        }
    }
    inline static uint32_t getBufferSize()
    {
        char* envBufferSize = getenv("CUPROFILE_BUFFER_SIZE");
        if (envBufferSize == nullptr) return BUFFER_SIZE_DEFAULT;
        return static_cast<uint32_t>(std::stoi(envBufferSize));
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
        uint32_t bufferSize = cupr::getBufferSize();

        cudaMalloc((void**) &context->deviceAccessRecords, sizeof(cupr::AccessRecord) * bufferSize);
        cudaMalloc((void**) &context->deviceSharedBuffers, sizeof(cupr::AllocRecord) * bufferSize);

        const uint32_t zero = 0;
        CHECK_CUDA_CALL(cudaMemcpyToSymbol(cupr::deviceAccessRecords, &context->deviceAccessRecords, sizeof(context->deviceAccessRecords)));
        CHECK_CUDA_CALL(cudaMemcpyToSymbol(cupr::deviceAccessRecordIndex, &zero, sizeof(zero)));
        CHECK_CUDA_CALL(cudaMemcpyToSymbol(cupr::deviceSharedBuffers, &context->deviceSharedBuffers, sizeof(context->deviceSharedBuffers)));
        CHECK_CUDA_CALL(cudaMemcpyToSymbol(cupr::deviceSharedBufferIndex, &zero, sizeof(zero)));
        CHECK_CUDA_CALL(cudaMemcpyToSymbol(cupr::deviceBufferSize, &bufferSize, sizeof(bufferSize)));

        context->timer->start();
    }
    void CU_PREFIX(kernelEnd)(cupr::KernelContext* context)
    {
        context->timer->stop_wait();
        CHECK_CUDA_CALL(cudaDeviceSynchronize());

        uint32_t accessCount = 0, sharedBuffersCount = 0;;
        CHECK_CUDA_CALL(cudaMemcpyFromSymbol(&accessCount, cupr::deviceAccessRecordIndex, sizeof(uint32_t)));
        CHECK_CUDA_CALL(cudaMemcpyFromSymbol(&sharedBuffersCount, cupr::deviceSharedBufferIndex, sizeof(uint32_t)));

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

        for (auto& alloc: cupr::allocations)
        {
            sharedBuffers.push_back(alloc);
        }

        emitKernelData(context->kernelName, records, sharedBuffers, context->timer->get_time());
    }
    void CU_PREFIX(malloc)(void* address, size_t size, size_t elementSize, const char* type)
    {
        cupr::allocations.emplace_back(address, size, elementSize, cupr::AddressSpace::Global, type);
    }
    void CU_PREFIX(free)(void* address)
    {
        for (auto& alloc: cupr::allocations)
        {
            if (alloc.address == address)
            {
                alloc.active = false;
            }
        }
    }

    static __forceinline__ __device__ uint32_t warpid()
    {
        uint32_t ret;
        asm volatile ("mov.u32 %0, %%warpid;" : "=r"(ret));
        return ret;
    }
    extern "C" __device__ void CU_PREFIX(store)(void* address, size_t size, uint32_t addressSpace,
                                                size_t type, int32_t debugIndex)
    {
        ATOMIC_INSERT(cupr::deviceAccessRecords,
                      &cupr::deviceAccessRecordIndex,
                      cupr::deviceBufferSize,
                      cupr::AccessRecord(cupr::AccessType::Write, blockIdx, threadIdx, warpid(),
                                         address, size, static_cast<cupr::AddressSpace>(addressSpace),
                                         static_cast<int64_t>(clock64()), type, debugIndex));
    }
    extern "C" __device__ void CU_PREFIX(load)(void* address, size_t size, uint32_t addressSpace,
                                               size_t type, int32_t debugIndex)
    {
        ATOMIC_INSERT(cupr::deviceAccessRecords,
                      &cupr::deviceAccessRecordIndex,
                      cupr::deviceBufferSize,
                      cupr::AccessRecord(cupr::AccessType::Read, blockIdx, threadIdx, warpid(),
                              address, size, static_cast<cupr::AddressSpace>(addressSpace),
                              static_cast<int64_t>(clock64()), type, debugIndex));
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
        ATOMIC_INSERT(cupr::deviceSharedBuffers,
                      &cupr::deviceSharedBufferIndex,
                      cupr::deviceBufferSize,
                      cupr::AllocRecord(address, size, elementSize, cupr::AddressSpace::Shared, type));
    }
}
