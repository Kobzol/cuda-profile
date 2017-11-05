#include "Memtracker.h"
#include "../RuntimeState.h"

using namespace cupr;

bool CU_PREFIX(isRuntimeTrackingEnabled)()
{
    return false;
}

namespace cupr {
    MemTracker memTracker;
}

void MemTracker::malloc(void* address, size_t size, size_t elementSize, const char* type,
                       const char* name, const char* location)
{
    cupr::state.getAllocations().emplace_back(address, size, elementSize, cupr::AddressSpace::Global,
                                              type, name, location);
}
void MemTracker::free(void* address)
{
    for (auto& alloc: cupr::state.getAllocations())
    {
        if (alloc.address == address)
        {
            alloc.active = false;
        }
    }
}
