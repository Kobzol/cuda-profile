#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

#include "../Utility.h"
#include "../tracedata/AccessRecord.h"
#include "../tracedata/AllocRecord.h"
#include "../tracedata/AddressSpace.h"
#include "../DeviceDimensions.h"

namespace cupr
{
    class TraceFormatter
    {
    public:
        TraceFormatter() = default;
        virtual ~TraceFormatter() = default;

        virtual void formatTrace(std::ostream& os,
                                 const std::string& kernel,
                                 DeviceDimensions dimensions,
                                 const std::vector<AccessRecord>& accesses,
                                 const std::vector<AllocRecord>& allocations,
                                 double start,
                                 double end,
                                 bool prettify) = 0;
        virtual std::string getSuffix() = 0;
    protected:
        std::string hexPointer(const void* ptr);
    };
}
