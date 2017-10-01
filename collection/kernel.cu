#include "general.h"
#include "device/CuprRuntime.h"

__global__ void kernel(int *p)
{
    *p = 5;
}

void cudaTest()
{
    const int COUNT = 32;

    int* dPtr;
    int data[COUNT] = { 0 };

    cudaMalloc((void**) &dPtr, sizeof(data));
    cudaMemcpy(dPtr, data, sizeof(data), cudaMemcpyHostToDevice);

    kernel<<<32, COUNT>>>(dPtr);

    int ptr[COUNT];
    cudaMemcpy(ptr, dPtr, sizeof(data), cudaMemcpyDeviceToHost);

    cudaFree(dPtr);
}
