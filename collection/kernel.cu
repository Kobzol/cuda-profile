#include "general.h"
#include "device/CuprRuntime.h"

__global__ void kernel(int *p)
{
    p[threadIdx.x] = 5;
}

void cudaTest()
{
    const int COUNT = 16;

    int* dPtr;
    int data[COUNT] = { 0 };

    cudaMalloc((void**) &dPtr, sizeof(data));
    cudaMemcpy(dPtr, data, sizeof(data), cudaMemcpyHostToDevice);

    kernel<<<2, COUNT>>>(dPtr);

    int ptr[COUNT];
    cudaMemcpy(ptr, dPtr, sizeof(data), cudaMemcpyDeviceToHost);

    cudaFree(dPtr);
}
