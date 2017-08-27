#include "general.h"
#include "runtime/Runtime.h"

__global__ void kernel(int *p)
{
    *p = 5;
    *p = 5;
    *p = 5;
}

void cudaTest()
{
    const int COUNT = 1;

    int* dPtr;
    int data[COUNT] = { 0 };

    cudaMalloc((void**) &dPtr, sizeof(data));
    cudaMemcpy(dPtr, data, sizeof(data), cudaMemcpyHostToDevice);

    kernel<<<1, COUNT>>>(dPtr);

    int ptr[COUNT];
    cudaMemcpy(ptr, dPtr, sizeof(data), cudaMemcpyDeviceToHost);

    cudaFree(dPtr);
}
