#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "../tracedata/AccessRecord.h"
#include "../tracedata/AllocRecord.h"

namespace cupr
{
    class Emitter
    {
    public:
        void initialize();
        void emitProgramRun();

        void emitKernelTrace(const std::string& kernelName,
                             const std::vector<AccessRecord>& records,
                             const std::vector<AllocRecord>& allocations,
                             float duration);

    private:
        void emitKernelTraceJson(const std::string& fileName,
                                 const std::string& kernel,
                                 const std::vector<AccessRecord>& records,
                                 const std::vector<AllocRecord>& allocations,
                                 double start,
                                 double end);
        void emitKernelTraceProtobuf(const std::string& fileName,
                                     const std::string& kernel,
                                     const std::vector<AccessRecord>& records,
                                     const std::vector<AllocRecord>& allocations,
                                     double start,
                                     double end);

        std::string generateDirectoryName();
        std::string getFilePath(const std::string& name);

        int kernelCounter = 0;
        int64_t timestampStart = getTimestamp();
        std::string directory;

        void copyMetadataFiles();
    };
}
