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
        void emitKernelData(const std::string& kernelName,
                            const std::vector<AccessRecord>& records,
                            const std::vector<AllocRecord>& allocations,
                            float duration);

    private:
        int kernelCounter = 0;
    };
}
