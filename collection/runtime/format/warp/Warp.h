#pragma once

#include <cstdint>
#include <cstddef>
#include <vector_types.h>
#include <vector>

namespace cupr {
    struct Access
    {
    public:
        Access() = default;
        Access(size_t address, size_t value, const uint3& threadIndex);

        size_t address;
        size_t value;
        uint3 threadIndex;
    };

    class Warp
    {
    public:
        Warp() = default;
        Warp(uint32_t id, size_t timestamp, const uint3& blockIndex,
             int32_t debugId, uint32_t typeId, uint8_t size, uint8_t accessType, uint8_t space);

        std::vector<Access> accesses;
        uint32_t id;
        size_t timestamp;
        uint3 blockIndex;
        int32_t debugId;
        uint32_t typeId;
        uint8_t size;
        uint8_t accessType;
        uint8_t space;
    };
}
