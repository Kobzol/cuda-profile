#include "Warp.h"

using namespace cupr;

Access::Access(size_t address, size_t value, const uint3& threadIndex)
        : address(address), value(value), threadIndex(threadIndex)
{

}

Warp::Warp(uint32_t id, size_t timestamp, const uint3& blockIndex, int32_t debugId, uint32_t typeId,
           uint8_t size, uint8_t accessType, uint8_t space) : id(id), timestamp(timestamp),
                                                              blockIndex(blockIndex), debugId(debugId), typeId(typeId),
                                                              size(size), accessType(accessType), space(space)
{

}
