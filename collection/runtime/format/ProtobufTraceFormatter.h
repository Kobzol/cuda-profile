#pragma once

#include "TraceFormatter.h"

namespace cupr
{
    class ProtobufTraceFormatter : public TraceFormatter
    {
    public:
        void formatTrace(std::ostream& os, const std::string& kernel, DeviceDimensions dimensions,
                         const std::vector<Warp>& warps, const std::vector<AllocRecord>& allocations,
                         double start, double end, bool prettify, bool compress) final;

        std::string getSuffix() final;
    };
}
