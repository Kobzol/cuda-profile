#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

#include "../Utility.h"
#include "../tracedata/AccessRecord.h"
#include "picojson.h"
#include "../tracedata/AllocRecord.h"
#include "../tracedata/AddressSpace.h"

namespace cupr
{
    class Formatter
    {
    public:
        void outputKernelTraceJson(std::ostream& os,
                                   const std::string& kernel,
                                   dim3 dimensions[2],
                                   const std::vector<AccessRecord>& accesses,
                                   const std::vector<AllocRecord>& allocations,
                                   double start,
                                   double end,
                                   bool prettify);

        void outputKernelTraceProtobuf(std::ostream& os,
                                       const std::string& kernel,
                                       dim3 dimensions[2],
                                       const std::vector<AccessRecord>& accesses,
                                       const std::vector<AllocRecord>& allocations,
                                       double start,
                                       double end);

        void outputProgramRun(std::fstream& os, int64_t timestampStart, int64_t timestampEnd);

    private:
        std::string hexPointer(const void* ptr);

        picojson::value jsonify(const AccessRecord& record);
        picojson::value jsonify(const AllocRecord& record);

        template<typename T>
        picojson::value jsonify(const std::vector<T>& items)
        {
            std::vector<picojson::value> jsonified;
            for (auto& item: items)
            {
                jsonified.push_back(this->jsonify(item));
            }

            return picojson::value(jsonified);
        }
    };
}
