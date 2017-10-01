#pragma once

#include "tracedata/AllocRecord.h"
#include "format/Emitter.h"

namespace cupr
{
    class RuntimeState
    {
    public:
        RuntimeState()
        {
            this->emitter.initialize();
        }

        std::vector<cupr::AllocRecord> allocations;
        Emitter emitter;
    };

    extern RuntimeState state;
}
