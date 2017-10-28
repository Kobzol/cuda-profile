#include "JsonTraceFormatter.h"

#include "zlib/zstr.hpp"

picojson::value cupr::JsonTraceFormatter::jsonify(const cupr::AccessRecord& record)
{
    return picojson::value(picojson::object{
            {"threadIdx", picojson::value(picojson::object {
                    {"x", picojson::value((double) record.threadIdx.x)},
                    {"y", picojson::value((double) record.threadIdx.y)},
                    {"z", picojson::value((double) record.threadIdx.z)}
            })},
            {"blockIdx",  picojson::value(picojson::object {
                    {"x", picojson::value((double) record.blockIdx.x)},
                    {"y", picojson::value((double) record.blockIdx.y)},
                    {"z", picojson::value((double) record.blockIdx.z)}
            })},
            {"warpId",    picojson::value((double) record.warpId)},
            {"debugId",   picojson::value((double) record.debugIndex)},
            {"address",   picojson::value(this->hexPointer(record.address))},
            {"kind",      picojson::value((double) (record.kind == AccessType::Read ? 0 : 1))},
            {"size",      picojson::value((double) record.size)},
            {"space",     picojson::value((double) record.addressSpace)},
            {"typeIndex", picojson::value((double) record.type)},
            {"timestamp", picojson::value((double) record.timestamp)}
    });
}

picojson::value cupr::JsonTraceFormatter::jsonify(const cupr::AllocRecord& record)
{
    std::string typeKey = "typeString";
    picojson::value typeValue;
    if (record.type == nullptr)
    {
        typeKey = "typeIndex";
        typeValue = picojson::value((double) record.typeIndex);
    } else typeValue = picojson::value(record.type);

    return picojson::value(picojson::object {
            {"address",     picojson::value(this->hexPointer(record.address))},
            {"size",        picojson::value((double) record.size)},
            {"elementSize", picojson::value((double) record.elementSize)},
            {"space",       picojson::value((double) record.addressSpace)},
            {typeKey,       typeValue},
            {"active",      picojson::value(record.active)}
    });
}

void cupr::JsonTraceFormatter::formatTrace(std::ostream& os,
                                           const std::string& kernel,
                                           DeviceDimensions dimensions,
                                           const std::vector<cupr::AccessRecord>& accesses,
                                           const std::vector<cupr::AllocRecord>& allocations,
                                           double start,
                                           double end,
                                           bool prettify,
                                           bool compress)
{
    auto value = picojson::value(picojson::object {
            {"type", picojson::value("trace")},
            {"kernel", picojson::value(kernel)},
            {"allocations",  this->jsonify(allocations)},
            {"accesses",   this->jsonify(accesses)},
            {"start", picojson::value(start)},
            {"end", picojson::value(end)},
            {"gridDim",  picojson::value(picojson::object {
                    {"x", picojson::value((double) dimensions.grid.x)},
                    {"y", picojson::value((double) dimensions.grid.y)},
                    {"z", picojson::value((double) dimensions.grid.z)}
            })},
            {"blockDim",  picojson::value(picojson::object {
                    {"x", picojson::value((double) dimensions.block.x)},
                    {"y", picojson::value((double) dimensions.block.y)},
                    {"z", picojson::value((double) dimensions.block.z)}
            })},
            {"warpSize", picojson::value((double) dimensions.warpSize)}
    });

    if (compress)
    {
        zstr::ostream compressed(os);
        compressed << value.serialize(prettify);
    }
    else os << value.serialize(prettify);
}

std::string cupr::JsonTraceFormatter::getSuffix()
{
    return "json";
}
