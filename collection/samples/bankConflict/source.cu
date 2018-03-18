#include <CuprRuntime.h>
#include <cuda_runtime_api.h>

__global__ void kernel(int *p, int *q)
{
    __shared__ float mem[64];

    int index = 0;
    if (threadIdx.x < 16)
    {
        index = threadIdx.x;
    }
    else if (threadIdx.x < 20)
    {
        index = threadIdx.x + 16;
    }
    else index = threadIdx.x;

    mem[index] = threadIdx.x;
}

int main(int argc, char** argv)
{
    const int COUNT = 32;

    int* dPtr;
    int data[COUNT] = { 0 };

    cudaMalloc((void**) &dPtr, sizeof(data));
    cudaMemcpy(dPtr, data, sizeof(data), cudaMemcpyHostToDevice);

    kernel<<<1, COUNT>>>(dPtr, dPtr);

    int ptr[COUNT];
    cudaMemcpy(ptr, dPtr, sizeof(data), cudaMemcpyDeviceToHost);

    cudaFree(dPtr);

    return 0;
}
