#pragma once

#include <cstddef>

#include "../Prefix.h"

extern "C" bool CU_PREFIX(isRuntimeTrackingEnabled)();

namespace cupr {
    class MemTracker
    {
    public:
        void malloc(void* address, size_t size, size_t elementSize, const char* type,
                    const char* name, const char* location);
        void free(void* address);
    };

    extern MemTracker memTracker;
}
