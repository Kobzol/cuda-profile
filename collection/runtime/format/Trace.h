#pragma once

#include <string>
#include "../DeviceDimensions.h"
#include "../tracedata/AccessRecord.h"
#include "../tracedata/AllocRecord.h"

namespace cupr {
    class Trace
    {
    public:
        Trace(std::string kernelName, DeviceDimensions dimensions, AccessRecord* records,
              size_t recordCount, const std::vector<AllocRecord>& allocations, float duration);

        std::string kernelName;
        DeviceDimensions dimensions;
        AccessRecord* records;
        size_t recordCount;
        std::vector<AllocRecord> allocations;
        float duration;
    };
}
