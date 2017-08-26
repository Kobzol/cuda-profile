#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

#include "cudautil.h"
#include "AccessRecord.h"
#include "picojson.h"
#include "AllocRecord.h"
#include "AddressSpace.h"

class Formatter
{
public:
    picojson::value jsonify(const AccessRecord& record)
    {
        return picojson::value(picojson::object{
                {"threadIdx", picojson::value(picojson::object {
                        {"x", picojson::value((double) record.threadIdx.x)},
                        {"y", picojson::value((double) record.threadIdx.y)},
                        {"z", picojson::value((double) record.threadIdx.z)}
                })},
                {"blockIdx", picojson::value(picojson::object {
                        {"x", picojson::value((double) record.blockIdx.x)},
                        {"y", picojson::value((double) record.blockIdx.y)},
                        {"z", picojson::value((double) record.blockIdx.z)}
                })},
                {"warpId", picojson::value((double) record.warpId)},
                {"debugId", picojson::value((double) record.debugIndex)},
                {"event", picojson::value(picojson::object {
                        {"address", picojson::value(this->hexPointer(record.address))},
                        {"kind", picojson::value((record.accessType == AccessType::Read ? "read" : "write"))},
                        {"size", picojson::value((double) record.size)},
                        {"space", picojson::value(this->getAddressSpace(record.addressSpace))},
                        {"type", picojson::value(record.type)},
                        {"timestamp", picojson::value((double) record.timestamp)}
                })}
        });
    }
    picojson::value jsonify(const AllocRecord& record)
    {
        return picojson::value(picojson::object {
                {"address", picojson::value(this->hexPointer(record.address))},
                {"size", picojson::value((double) record.size)},
                {"elementSize", picojson::value((double) record.elementSize)},
                {"space", picojson::value(this->getAddressSpace(record.addressSpace))},
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
            jsonified.push_back(this->jsonify(item));
        }

        return picojson::value(jsonified);
    }

    void outputKernelRun(std::ostream& os,
                         const std::vector<AccessRecord>& accesses,
                         const std::vector<AllocRecord>& allocations,
                         float kernelTime)
    {
        auto value = picojson::value(picojson::object {
                {"memoryMap", this->jsonify(allocations)},
                {"accesses", this->jsonify(accesses)},
                {"kernelTime", picojson::value(kernelTime)}
        });

        os << value.serialize(true);
    }

private:
    std::string hexPointer(const void* ptr)
    {
        std::ostringstream address;
        address << ptr;
        return address.str();
    }

    std::string getAddressSpace(AddressSpace space)
    {
        switch (space)
        {
            case AddressSpace::Shared: return "shared";
            case AddressSpace::Constant: return "constant";
            default: return "global";
        }
    }
};
