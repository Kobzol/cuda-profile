#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "AccessRecord.h"
#include "AllocRecord.h"

namespace cupr
{
    class Emitter
    {
    public:
        void initialize();
        void emitProgramRun();

        void emitKernelData(const std::string& kernelName,
                            const std::vector<AccessRecord>& records,
                            const std::vector<AllocRecord>& allocations,
                            float duration);

    private:
        void emitKernelDataJson(const std::string& fileName,
                                const std::string& kernel,
                                const std::vector<AccessRecord>& records,
                                const std::vector<AllocRecord>& allocations,
                                float duration,
                                int64_t timestamp);
        void emitKernelDataProtobuf(const std::string& fileName,
                                    const std::string& kernel,
                                    const std::vector<AccessRecord>& records,
                                    const std::vector<AllocRecord>& allocations,
                                    float duration,
                                    int64_t timestamp);

        std::string generateDirectoryName();
        std::string getFilePath(const std::string& name);

        int kernelCounter = 0;
        int64_t timestampStart = getTimestamp();
        std::string directory;

        void copyMetadataFiles();
    };
}
