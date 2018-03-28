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
#include "warp/Warp.h"

namespace cupr
{
    class TraceFormatter
    {
    public:
        TraceFormatter()
        {
            this->hexString.resize(18);
            this->hexString[0] = '0';
            this->hexString[1] = 'x';
        }
        virtual ~TraceFormatter() = default;

        virtual void formatTrace(std::ostream& os,
                                 const std::string& kernel,
                                 DeviceDimensions dimensions,
                                 const std::vector<Warp>& warps,
                                 const std::vector<AllocRecord>& allocations,
                                 double start,
                                 double end,
                                 bool prettify,
                                 bool compress) = 0;
        virtual std::string getSuffix() = 0;
        virtual bool isBinary()
        {
            return true;
        }
        virtual bool supportsCompression()
        {
            return true;
        }
    protected:
        std::string hexPointer(const void* ptr);

        std::string hexString;
    };
}
