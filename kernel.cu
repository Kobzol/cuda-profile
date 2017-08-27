#include "general.h"
#include "runtime/Runtime.h"

__global__ void kernel(int *p)
{
    for (int i = 0; i < 10; i++)
    {
        p[i] = p[i + 1];
    }
}

void cudaTest()
{
    const int COUNT = 64;

    int* dPtr;
    int data[COUNT] = { 0 };

    cudaMalloc((void**) &dPtr, sizeof(data));
    cudaMemcpy(dPtr, data, sizeof(data), cudaMemcpyHostToDevice);

    kernel<<<32, COUNT>>>(dPtr);

    int ptr[COUNT];
    cudaMemcpy(ptr, dPtr, sizeof(data), cudaMemcpyDeviceToHost);

    cudaFree(dPtr);
}
