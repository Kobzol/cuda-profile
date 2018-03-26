#pragma once

#include "TraceFormatter.h"
#include "json/picojson.h"

namespace cupr
{
    class JsonTraceFormatter: public TraceFormatter
    {
    public:
        void formatTrace(std::ostream& os, const std::string& kernel, DeviceDimensions dimensions,
                         const std::vector<Warp>& warps, const std::vector<AllocRecord>& allocations,
                         double start, double end, bool prettify, bool compress) final;

        std::string getSuffix() final;
        bool isBinary() final
        {
            return false;
        }
        bool supportsCompression() final
        {
            return true;
        }

    private:
        picojson::value jsonify(const Warp& warp);
        picojson::value jsonify(const Access& record);
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
