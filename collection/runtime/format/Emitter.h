#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "../tracedata/AccessRecord.h"
#include "../tracedata/AllocRecord.h"
#include "TraceFormatter.h"

namespace cupr
{
    class Emitter
    {
    public:
        Emitter(std::unique_ptr<TraceFormatter> formatter, bool prettify, bool compress);

        void emitProgramRun();
        void emitKernelTrace(const std::string& kernelName, const DeviceDimensions& dimensions, AccessRecord* records,
                             size_t recordCount, const std::vector<AllocRecord>& allocations, float duration);

    private:
        std::string generateDirectoryName();
        std::string getFilePath(const std::string& name);

        void copyMetadataFiles();
        std::string getTraceSuffix();

        std::unique_ptr<TraceFormatter> formatter;
        bool prettify;
        bool compress;

        int64_t timestampStart = getTimestamp();
        std::string directory;

        std::unordered_map<std::string, int> kernelCount;
        std::vector<std::string> traceFiles;
    };
}
