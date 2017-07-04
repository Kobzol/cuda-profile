#include "general.h"
#include "runtime/Runtime.h"

__global__ void kernel(int* p)
{
    p[threadIdx.x] = threadIdx.x;
}

void cudaTest()
{
    __cu_initMemory();

    const int COUNT = 10;

    int* dPtr;
    int data[COUNT] = { 0 };

    cudaMalloc((void**) &dPtr, sizeof(data));
    cudaMemcpy(dPtr, data, sizeof(data), cudaMemcpyHostToDevice);

    __cu_kernelStart();
    kernel<<<1, COUNT>>>(dPtr);
    __cu_kernelEnd();

    int ptr[COUNT];
    cudaMemcpy(ptr, dPtr, sizeof(data), cudaMemcpyDeviceToHost);
}
