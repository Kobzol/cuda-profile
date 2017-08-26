#pragma context

#include <string>
#include <utility>

#include "AccessRecord.h"
#include "CudaTimer.h"
#include "AllocRecord.h"

struct KernelContext
{
public:
    const char* kernelName;
    AccessRecord* deviceAccessRecords;
    AllocRecord* deviceSharedBuffers;
    CudaTimer* timer;
};
