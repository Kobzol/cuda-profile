#include "CudaTimer.h"

cupr::CudaTimer::CudaTimer(bool automatic) : automatic(automatic)
{
    CHECK_CUDA_CALL(cudaEventCreate(&this->startEvent));
    CHECK_CUDA_CALL(cudaEventCreate(&this->stopEvent));

    if (automatic)
    {
        this->start();
    }
}

cupr::CudaTimer::~CudaTimer()
{
    if (this->automatic)
    {
        this->stop_wait();
    }

    CHECK_CUDA_CALL(cudaEventDestroy(this->startEvent));
    CHECK_CUDA_CALL(cudaEventDestroy(this->stopEvent));
}

void cupr::CudaTimer::start() const
{
    CHECK_CUDA_CALL(cudaEventRecord(this->startEvent));
}

void cupr::CudaTimer::stop_wait() const
{
    CHECK_CUDA_CALL(cudaEventRecord(this->stopEvent));
    CHECK_CUDA_CALL(cudaEventSynchronize(this->stopEvent));
}

float cupr::CudaTimer::get_time() const
{
    float time;
    CHECK_CUDA_CALL(cudaEventElapsedTime(&time, this->startEvent, this->stopEvent));
    return time;
}
