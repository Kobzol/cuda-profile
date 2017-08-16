#include "general.h"
#include "runtime/Runtime.h"

__global__ void kernel(int* p)
{
    p[threadIdx.x] = threadIdx.x;
}

void cudaTest()
{
    const int COUNT = 64;

    int* dPtr;
    int data[COUNT] = { 0 };

    cudaMalloc((void**) &dPtr, sizeof(data));
    cudaMemcpy(dPtr, data, sizeof(data), cudaMemcpyHostToDevice);

    kernel<<<1, COUNT>>>(dPtr);

    int ptr[COUNT];
    cudaMemcpy(ptr, dPtr, sizeof(data), cudaMemcpyDeviceToHost);

    cudaFree(dPtr);
}
