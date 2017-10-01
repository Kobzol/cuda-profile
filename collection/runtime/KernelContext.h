#pragma context

#include <string>
#include <utility>

#include "tracedata/AccessRecord.h"
#include "CudaTimer.h"
#include "tracedata/AllocRecord.h"

namespace cupr
{
    struct KernelContext
    {
    public:
        const char* kernelName;
        AccessRecord* deviceAccessRecords;
        AllocRecord* deviceSharedBuffers;
        CudaTimer* timer;
    };
}
