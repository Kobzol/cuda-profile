#include "WarpGrouper.h"

#include <unordered_map>
#include <algorithm>

using namespace cupr;

struct WarpId
{
public:
    WarpId(size_t blockId, size_t warpId, size_t timestamp)
        : blockId(blockId), warpId(warpId), timestamp(timestamp)
    {

    }

    size_t blockId;
    size_t warpId;
    size_t timestamp;
};

bool operator==(const WarpId& lhs, const WarpId& rhs)
{
    return lhs.blockId == rhs.blockId &&
           lhs.warpId == rhs.warpId &&
           lhs.timestamp == rhs.timestamp;
}

namespace std {
    #define OFFSET_BASIS 2166136261ul
    #define FNV_PRIME 16777619ul

    template <> struct hash<WarpId>
    {
        size_t operator()(const WarpId& id) const
        {
            auto i = id.blockId;
            auto j = id.warpId;
            auto k = id.timestamp;
            return ((((((OFFSET_BASIS ^ i) * FNV_PRIME) ^ j) * FNV_PRIME) ^ k) * FNV_PRIME);
        }
    };
}

size_t getBlockId(const uint3& block, const DeviceDimensions& dimensions)
{
    return block.z * (dimensions.block.x * dimensions.block.y) + block.y * dimensions.block.x + dimensions.block.y;
}

std::vector<Warp> WarpGrouper::groupWarps(const std::vector<AccessRecord>& records, const DeviceDimensions& dimensions)
{
    std::unordered_map<WarpId, Warp> warpMap;

    for (const auto& access : records)
    {
        WarpId key(getBlockId(access.blockIdx, dimensions), access.warpId, static_cast<size_t>(access.timestamp));
        auto it = warpMap.find(key);
        if (it == warpMap.end())
        {
            it = warpMap.insert(std::make_pair(key, Warp(
                    access.warpId,
                    static_cast<size_t>(access.timestamp),
                    access.blockIdx,
                    access.debugIndex,
                    static_cast<uint32_t>(access.type),
                    static_cast<uint8_t>(access.size),
                    static_cast<uint8_t>(access.kind),
                    static_cast<uint8_t>(access.addressSpace)
            ))).first;
            it->second.accesses.reserve(dimensions.warpSize);
        }

        it->second.accesses.emplace_back(reinterpret_cast<size_t>(access.address), access.value, access.threadIdx);
    }

    std::vector<Warp> warps;
    warps.reserve(warpMap.size());
    for (auto& kv: warpMap)
    {
        warps.push_back(kv.second);
    }

    std::sort(warps.begin(), warps.end(), [](const Warp& lhs, const Warp& rhs) {
        return lhs.timestamp < rhs.timestamp;
    });

    return warps;
}
