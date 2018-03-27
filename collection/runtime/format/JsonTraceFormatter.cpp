#include "JsonTraceFormatter.h"

#include "json/rapidjson/ostreamwrapper.h"
#include "json/rapidjson/prettywriter.h"

#ifdef CUPR_USE_ZLIB
    #include "zlib/zstr.hpp"
#endif

using namespace rapidjson;

void cupr::JsonTraceFormatter::formatTrace(std::ostream& os,
                                           const std::string& kernel,
                                           DeviceDimensions dimensions,
                                           const std::vector<cupr::Warp>& warps,
                                           const std::vector<cupr::AllocRecord>& allocations,
                                           double start,
                                           double end,
                                           bool prettify,
                                           bool compress)
{
    auto output = this->createStream(os, compress);

    OStreamWrapper stream(*output);
    auto writer = this->createWriter(prettify, stream);

    this->jsonify(*writer, kernel, dimensions, warps, allocations, start, end);
}

std::string cupr::JsonTraceFormatter::getSuffix()
{
    return "json";
}

template<typename Stream>
std::unique_ptr<rapidjson::Writer<Stream>> cupr::JsonTraceFormatter::createWriter(bool prettify,
                                                                                  Stream& stream)
{
    if (prettify)
    {
        return std::make_unique<PrettyWriter<Stream>>(stream);
    }
    else return std::make_unique<Writer<Stream>>(stream);
}

std::unique_ptr<std::ostream> cupr::JsonTraceFormatter::createStream(std::ostream& input, bool compress)
{
#ifdef CUPR_USE_ZLIB
    if (compress)
    {
        return std::make_unique<zstr::ostream>(input);
    }
#endif
    return std::make_unique<std::ostream>(input.rdbuf());
}

template<typename Stream>
void cupr::JsonTraceFormatter::jsonify(rapidjson::Writer<Stream>& writer, const uint3& dim)
{
    writer.StartObject();
    writer.String("x");
    writer.Int(dim.x);
    writer.String("y");
    writer.Int(dim.y);
    writer.String("z");
    writer.Int(dim.z);
    writer.EndObject();
}

template<typename Stream, typename T>
void cupr::JsonTraceFormatter::jsonify(rapidjson::Writer<Stream>& writer, const std::vector<T>& items)
{
    writer.StartArray();
    for (auto& item: items)
    {
        this->jsonify(writer, item);
    }
    writer.EndArray();
}

template<typename Stream>
void cupr::JsonTraceFormatter::jsonify(rapidjson::Writer<Stream>& writer, const cupr::Warp& warp)
{
    writer.StartObject();

    writer.String("accesses");
    this->jsonify(writer, warp.accesses);

    writer.String("blockIdx");
    this->jsonify(writer, warp.blockIndex);

    writer.String("warpId");
    writer.Int(warp.id);

    writer.String("debugId");
    writer.Int(warp.debugId);

    writer.String("kind");
    writer.Int(warp.accessType == static_cast<uint8_t>(cupr::AccessType::Read) ? 0 : 1);

    writer.String("size");
    writer.Int(warp.size);

    writer.String("space");
    writer.Int(warp.space);

    writer.String("typeIndex");
    writer.Int(warp.typeId);

    writer.String("timestamp");
    writer.String(std::to_string(warp.timestamp));

    writer.EndObject();
}

template<typename Stream>
void cupr::JsonTraceFormatter::jsonify(rapidjson::Writer<Stream>& writer, const cupr::Access& record)
{
    writer.StartObject();

    writer.String("threadIdx");
    this->jsonify(writer, record.threadIndex);

    writer.String("address");
    writer.String(this->hexPointer(reinterpret_cast<const void*>(record.address)));

    writer.String("value");
    writer.String(this->hexPointer(reinterpret_cast<const void*>(record.value)));

    writer.EndObject();
}

template<typename Stream>
void cupr::JsonTraceFormatter::jsonify(rapidjson::Writer<Stream>& writer, const cupr::AllocRecord& record)
{
    writer.StartObject();

    if (record.type == nullptr)
    {
        writer.String("typeIndex");
        writer.Int(record.typeIndex);
    }
    else
    {
        writer.String("typeString");
        writer.String(record.type);
    }

    if (record.name == nullptr)
    {
        writer.String("nameIndex");
        writer.Int(record.nameIndex);
    }
    else
    {
        writer.String("nameString");
        writer.String(record.name);
    }

    writer.String("address");
    writer.String(this->hexPointer(record.address));

    writer.String("size");
    writer.Int(static_cast<unsigned int>(record.size));

    writer.String("elementSize");
    writer.Int(static_cast<int>(record.elementSize));

    writer.String("space");
    writer.Int(static_cast<int>(record.addressSpace));

    writer.String("active");
    writer.Bool(record.active);

    writer.String("location");
    writer.String(record.location == nullptr ? "" : record.location);

    writer.EndObject();
}

template<typename Stream>
void cupr::JsonTraceFormatter::jsonify(rapidjson::Writer<Stream>& writer, const std::string& kernel,
                                       DeviceDimensions dimensions, const std::vector<cupr::Warp>& warps,
                                       const std::vector<cupr::AllocRecord>& allocations, double start, double end)
{
    writer.StartObject();

    writer.String("type");
    writer.String("trace");

    writer.String("kernel");
    writer.String(kernel);

    writer.String("allocations");
    this->jsonify(writer, allocations);

    writer.String("warps");
    this->jsonify(writer, warps);

    writer.String("start");
    writer.Double(start);

    writer.String("end");
    writer.Double(end);

    writer.String("gridDim");
    this->jsonify(writer, dimensions.grid);

    writer.String("blockDim");
    this->jsonify(writer, dimensions.block);

    writer.String("warpSize");
    writer.Int(dimensions.warpSize);

    writer.String("bankSize");
    writer.Int(dimensions.bankSize);

    writer.EndObject();
}
