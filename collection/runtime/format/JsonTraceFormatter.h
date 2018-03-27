#pragma once

#include <memory>
#include "TraceFormatter.h"

#define RAPIDJSON_HAS_STDSTRING 1
#include "json/rapidjson/writer.h"

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
        template <typename Stream>
        std::unique_ptr<rapidjson::Writer<Stream>> createWriter(bool prettify, Stream& stream);
        std::unique_ptr<std::ostream> createStream(std::ostream& input, bool compress);

        template <typename Stream>
        void jsonify(rapidjson::Writer<Stream>& writer, const uint3& dim);

        template <typename Stream, typename T>
        void jsonify(rapidjson::Writer<Stream>& writer, const std::vector<T>& items);

        template <typename Stream>
        void jsonify(rapidjson::Writer<Stream>& writer, const cupr::Warp& warp);

        template <typename Stream>
        void jsonify(rapidjson::Writer<Stream>& writer, const cupr::Access& record);

        template <typename Stream>
        void jsonify(rapidjson::Writer<Stream>& writer, const cupr::AllocRecord& record);

        template <typename Stream>
        void jsonify(rapidjson::Writer<Stream>& writer, const std::string& kernel,
                     DeviceDimensions dimensions,
                     const std::vector<cupr::Warp>& warps,
                     const std::vector<cupr::AllocRecord>& allocations,
                     double start,
                     double end);
    };
}
