#pragma once

#include "TraceFormatter.h"

namespace cupr
{
    class ProtobufTraceFormatter : public TraceFormatter
    {
    public:
        void formatTrace(std::ostream& os, const std::string& kernel, DeviceDimensions dimensions,
                         const std::vector<AccessRecord>& accesses, const std::vector<AllocRecord>& allocations,
                         double start, double end, bool prettify, bool compress) override;

        std::string getSuffix() override;
    };
}
