#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

#include "Utility.h"
#include "AccessRecord.h"
#include "picojson.h"
#include "AllocRecord.h"
#include "AddressSpace.h"

namespace cupr
{
    class Formatter
    {
    public:
        void outputKernelRunJson(std::ostream& os,
                                 const std::string& kernel,
                                 const std::vector<AccessRecord>& accesses,
                                 const std::vector<AllocRecord>& allocations,
                                 float duration,
                                 int64_t timestamp,
                                 bool prettify);

        void outputKernelRunProtobuf(std::ostream& os,
                                     const std::string& kernel,
                                     const std::vector<AccessRecord>& accesses,
                                     const std::vector<AllocRecord>& allocations,
                                     float duration,
                                     int64_t timestamp);

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
