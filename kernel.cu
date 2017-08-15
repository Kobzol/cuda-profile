#include "general.h"
#include "runtime/Runtime.h"

__global__ void kernel(int* p, int p2)
{
    p[threadIdx.x] = threadIdx.x;
}

void cudaTest()
{
    const int COUNT = 10;

    int* dPtr;
    int data[COUNT] = { 0 };

    cudaMalloc((void**) &dPtr, sizeof(data));
    cudaMemcpy(dPtr, data, sizeof(data), cudaMemcpyHostToDevice);

    kernel<<<1, COUNT>>>(dPtr, 0);
    kernel<<<1, 5>>>(dPtr, 0);

    int ptr[COUNT];
    cudaMemcpy(ptr, dPtr, sizeof(data), cudaMemcpyDeviceToHost);
}
