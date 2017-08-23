#pragma context

#include <string>
#include <utility>

#include "AccessRecord.h"
#include "CudaTimer.h"

struct KernelContext
{
public:
    const char* kernelName;
    AccessRecord* deviceRecords;
    CudaTimer* timer;
};
