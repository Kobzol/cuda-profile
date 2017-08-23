#pragma once

#include "cudautil.h"


class CudaTimer
{
public:
    explicit CudaTimer(bool automatic = false) : automatic(automatic)
    {
        CHECK_CUDA_CALL(cudaEventCreate(&this->startEvent));
        CHECK_CUDA_CALL(cudaEventCreate(&this->stopEvent));

        if (automatic)
        {
            this->start();
        }
    }
    ~CudaTimer()
    {
        if (this->automatic)
        {
            this->stop_wait();
        }

        CHECK_CUDA_CALL(cudaEventDestroy(this->startEvent));
        CHECK_CUDA_CALL(cudaEventDestroy(this->stopEvent));
    }

    CudaTimer(const CudaTimer& other) = delete;
    CudaTimer& operator=(const CudaTimer& other) = delete;
    CudaTimer(CudaTimer&& other) = delete;

    void start() const
    {
        CHECK_CUDA_CALL(cudaEventRecord(this->startEvent));
    }
    void stop_wait() const
    {
        CHECK_CUDA_CALL(cudaEventRecord(this->stopEvent));
        CHECK_CUDA_CALL(cudaEventSynchronize(this->stopEvent));
    }
    float get_time() const
    {
        float time;
        CHECK_CUDA_CALL(cudaEventElapsedTime(&time, this->startEvent, this->stopEvent));
        return time;
    }

private:
    cudaEvent_t startEvent, stopEvent;
    bool automatic;
};
