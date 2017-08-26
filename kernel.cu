#include "general.h"
#include "runtime/Runtime.h"

__global__ void kernel()
{
    __shared__ int arr;
    printf("%p\n", &arr);
}

void cudaTest()
{
    const int COUNT = 2;

    int* dPtr;
    int data[COUNT] = { 0 };

    cudaMalloc((void**) &dPtr, sizeof(data));
    cudaMemcpy(dPtr, data, sizeof(data), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>();

    int ptr[COUNT];
    cudaMemcpy(ptr, dPtr, sizeof(data), cudaMemcpyDeviceToHost);

    cudaFree(dPtr);
}
