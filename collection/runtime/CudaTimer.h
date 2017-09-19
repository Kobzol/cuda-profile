#pragma once

#include "Utility.h"

namespace cupr
{
    class CudaTimer
    {
    public:
        explicit CudaTimer(bool automatic = false);
        ~CudaTimer();
        CudaTimer(const CudaTimer& other) = delete;
        CudaTimer& operator=(const CudaTimer& other) = delete;
        CudaTimer(CudaTimer&& other) = delete;
        CudaTimer& operator=(const CudaTimer&& other) = delete;

        void start() const;
        void stop_wait() const;
        float get_time() const;

    private:
        cudaEvent_t startEvent, stopEvent;
        bool automatic;
    };
}
