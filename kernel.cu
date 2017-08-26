#include "general.h"
#include "runtime/Runtime.h"

__constant__ int constArr[10] = { 0, 1 };

__global__ void kernel()
{
    __shared__ int arr[32];
    arr[5] = constArr[1];
}

void cudaTest()
{
    const int COUNT = 1;

    int* dPtr;
    int data[COUNT] = { 0 };

    cudaMalloc((void**) &dPtr, sizeof(data));
    cudaMemcpy(dPtr, data, sizeof(data), cudaMemcpyHostToDevice);

    kernel<<<1, COUNT>>>();

    int ptr[COUNT];
    cudaMemcpy(ptr, dPtr, sizeof(data), cudaMemcpyDeviceToHost);

    cudaFree(dPtr);
}
