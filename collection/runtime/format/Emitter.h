#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include "../tracedata/AccessRecord.h"
#include "../tracedata/AllocRecord.h"
#include "TraceFormatter.h"

namespace cupr
{
    class Emitter
    {
    public:
        Emitter(std::unique_ptr<TraceFormatter> formatter, bool prettify);
        void emitProgramRun();

        void emitKernelTrace(const std::string& kernelName,
                             const DeviceDimensions& dimensions,
                             const std::vector<AccessRecord>& records,
                             const std::vector<AllocRecord>& allocations,
                             float duration);

    private:
        std::string generateDirectoryName();
        std::string getFilePath(const std::string& name);

        void copyMetadataFiles();

        std::unique_ptr<TraceFormatter> formatter;
        bool prettify;

        int kernelCounter = 0;
        int64_t timestampStart = getTimestamp();
        std::string directory;
    };
}
