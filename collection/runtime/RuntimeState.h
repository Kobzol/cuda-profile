#pragma once

#include "tracedata/AllocRecord.h"
#include "format/Emitter.h"

namespace cupr
{
    class RuntimeState
    {
    public:
        RuntimeState();

        std::vector<AllocRecord>& getAllocations();
        Emitter& getEmitter();

    private:
        std::vector<cupr::AllocRecord> allocations;
        Emitter emitter;
    };

    extern RuntimeState state;
}
