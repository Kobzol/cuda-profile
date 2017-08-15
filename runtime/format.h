#pragma once

#include <iostream>
#include <fstream>
#include <vector>

#include "cudautil.h"
#include "AccessRecord.h"
#include "picojson.h"

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
    os << (record.accessType == AccessType::Read ? "read" : "write") << "[" << record.address;
    os << ", " << record.size << ", " << record.timestamp << "])";

    return os;
}

std::fstream& operator<<(std::fstream& fs, const AccessRecord& record)
{
    picojson::object root = {
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
            {"event", picojson::value({
                    {"address", picojson::value((double)((size_t) record.address))},
                    {"type", picojson::value((record.accessType == AccessType::Read ? "read" : "write"))},
                    {"size", picojson::value((double) record.size)},
                    {"timestamp", picojson::value((double) record.timestamp)}
            })}
    };

    picojson::value output(root);

    fs << output.serialize(true);

    return fs;
}

template <typename T>
std::fstream& operator<<(std::fstream& fs, const std::vector<T>& items)
{
    fs << "[";
    for (auto& item: items)
    {
        fs << item << ", ";
    }
    fs << "]" << std::endl;

    return fs;
}
