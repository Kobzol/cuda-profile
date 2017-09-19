#pragma once

#include "AllocRecord.h"
#include "Emitter.h"

namespace cupr
{
    class RuntimeState
    {
    public:
        std::vector<cupr::AllocRecord> allocations;
        Emitter emitter;
    };

    extern RuntimeState state;
}
