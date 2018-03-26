#include "JsonTraceFormatter.h"

#ifdef CUPR_USE_ZLIB
    #include "zlib/zstr.hpp"
#endif
/*
 *
 * */


picojson::value cupr::JsonTraceFormatter::jsonify(const cupr::Warp& warp)
{
    return picojson::value(picojson::object{
            {"accesses", this->jsonify(warp.accesses)},
            {"blockIdx",  picojson::value(picojson::object {
                    {"x", picojson::value((double) warp.blockIndex.x)},
                    {"y", picojson::value((double) warp.blockIndex.y)},
                    {"z", picojson::value((double) warp.blockIndex.z)}
            })},
            {"warpId",    picojson::value((double) warp.id)},
            {"debugId",   picojson::value((double) warp.debugId)},
            {"kind",      picojson::value((double) (warp.accessType == static_cast<uint8_t>(AccessType::Read) ? 0 : 1))},
            {"size",      picojson::value((double) warp.size)},
            {"space",     picojson::value((double) warp.space)},
            {"typeIndex", picojson::value((double) warp.typeId)},
            {"timestamp", picojson::value(std::to_string(warp.timestamp))},
    });
}

picojson::value cupr::JsonTraceFormatter::jsonify(const cupr::Access& record)
{
    return picojson::value(picojson::object{
            {"threadIdx", picojson::value(picojson::object {
                    {"x", picojson::value((double) record.threadIndex.x)},
                    {"y", picojson::value((double) record.threadIndex.y)},
                    {"z", picojson::value((double) record.threadIndex.z)}
            })},
            {"address",   picojson::value(this->hexPointer(reinterpret_cast<const void*>(record.address)))},
            {"value", picojson::value(this->hexPointer(reinterpret_cast<const void*>(record.value)))}
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
    }
    else typeValue = picojson::value(record.type);

    std::string nameKey = "nameString";
    picojson::value nameValue;
    if (record.name == nullptr)
    {
        nameKey = "nameIndex";
        nameValue = picojson::value((double) record.nameIndex);
    }
    else nameValue = picojson::value(record.name);

    return picojson::value(picojson::object {
            {"address",     picojson::value(this->hexPointer(record.address))},
            {"size",        picojson::value((double) record.size)},
            {"elementSize", picojson::value((double) record.elementSize)},
            {"space",       picojson::value((double) record.addressSpace)},
            {typeKey,       typeValue},
            {"active",      picojson::value(record.active)},
            {nameKey,       nameValue},
            {"location",    picojson::value(record.location == nullptr ? "" : record.location)},
    });
}

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
    auto value = picojson::value(picojson::object {
            {"type", picojson::value("trace")},
            {"kernel", picojson::value(kernel)},
            {"allocations",  this->jsonify(allocations)},
            {"warps",   this->jsonify(warps)},
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
            {"warpSize", picojson::value((double) dimensions.warpSize)},
            {"bankSize", picojson::value((double) dimensions.bankSize)}
    });

#ifdef CUPR_USE_ZLIB
    if (compress)
    {
        zstr::ostream compressed(os);
        compressed << value.serialize(prettify);
    }
    else os << value.serialize(prettify);
#else
    os << value.serialize(prettify);
#endif
}

std::string cupr::JsonTraceFormatter::getSuffix()
{
    return "json";
}
