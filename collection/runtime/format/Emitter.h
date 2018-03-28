#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include "../tracedata/AccessRecord.h"
#include "../tracedata/AllocRecord.h"
#include "TraceFormatter.h"
#include "Trace.h"
#include "worker/format-pool.h"

namespace cupr
{
    class Emitter
    {
    public:
        Emitter(std::unique_ptr<TraceFormatter> formatter, bool prettify, bool compress);

        void emitProgramRun();
        void emitKernelTrace(const std::string& kernelName, const DeviceDimensions& dimensions, AccessRecord* records,
                             size_t recordCount, const std::vector<AllocRecord>& allocations, float duration);

        void waitForJobs();

    private:
        void emitKernelTraceJob(std::unique_ptr<Trace> trace);

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

        std::mutex mutex;
        std::unique_ptr<FormatPool> pool;
        std::atomic<size_t> jobsProcessing{0};

        bool useThreadPool;
    };
}
