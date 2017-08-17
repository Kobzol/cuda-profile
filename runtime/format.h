#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

#include "cudautil.h"
#include "AccessRecord.h"
#include "picojson.h"
#include "AllocRecord.h"

std::ostream& operator<<(std::ostream& os, const dim3& dimension)
{
    os << dimension.x << ";" << dimension.y << ";" << dimension.z;
    return os;
}

std::ostream& operator<<(std::ostream& os, const AccessRecord& record)
{
    os << "Store(";
    os << record.blockIdx << ", ";
    os << record.threadIdx << ", ";
    os << record.warpId << ", ";
    os << (record.accessType == AccessType::Read ? "read" : "write") << "[" << record.address;
    os << ", " << record.size << ", " << record.timestamp << "])";

    return os;
}

std::string hexPointer(const void* ptr)
{
    std::ostringstream address;
    address << ptr;
    return address.str();
}
picojson::value jsonify(const AccessRecord& record)
{
    return picojson::value(picojson::object{
            {"threadIdx", picojson::value({
                    {"x", picojson::value((double) record.threadIdx.x)},
                    {"y", picojson::value((double) record.threadIdx.y)},
                    {"z", picojson::value((double) record.threadIdx.z)}
            })},
            {"blockIdx", picojson::value({
                    {"x", picojson::value((double) record.blockIdx.x)},
                    {"y", picojson::value((double) record.blockIdx.y)},
                    {"z", picojson::value((double) record.blockIdx.z)}
            })},
            {"warpId", picojson::value((double) record.warpId)},
            {"debugId", picojson::value((double) record.debugIndex)},
            {"event", picojson::value({
                    {"address", picojson::value(hexPointer(record.address))},
                    {"kind", picojson::value((record.accessType == AccessType::Read ? "read" : "write"))},
                    {"size", picojson::value((double) record.size)},
                    {"type", picojson::value(record.type)},
                    {"timestamp", picojson::value((double) record.timestamp)}
            })}
    });
}
picojson::value jsonify(const AllocRecord& record)
{
    return picojson::value(picojson::object {
            {"address", picojson::value(hexPointer(record.address))},
            {"size", picojson::value((double) record.size)},
            {"elementSize", picojson::value((double) record.elementSize)},
            {"type", picojson::value(record.type)},
            {"active", picojson::value(record.active)}
    });
}

template <typename T>
picojson::value jsonify(const std::vector<T>& items)
{
    std::vector<picojson::value> jsonified;
    for (auto& item: items)
    {
        jsonified.push_back(jsonify(item));
    }

    return picojson::value(jsonified);
}

void outputKernelRun(std::ostream& os, const std::vector<AccessRecord>& accesses,
                     const std::vector<AllocRecord>& allocations)
{
    auto value = picojson::value(picojson::object {
            {"memoryMap", jsonify(allocations)},
            {"accesses", jsonify(accesses)}
    });

    os << value.serialize(true);
}
