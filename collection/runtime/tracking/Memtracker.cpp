#include "Memtracker.h"
#include "../RuntimeState.h"

using namespace cupr;

void CU_PREFIX(malloc)(void* address, size_t size, size_t elementSize, const char* type,
                                  const char* name, const char* location)
{
    AllocRecord record(address, size, elementSize, cupr::AddressSpace::Global, type, name, location);

    for (auto& allocated: cupr::state.getAllocations())
    {
        if (allocated.address == record.address && allocated.active)
        {
            allocated = record;
            return;
        }
    }

    cupr::state.getAllocations().push_back(record);
}
void CU_PREFIX(free)(void* address)
{
    for (auto& alloc: cupr::state.getAllocations())
    {
        if (alloc.address == address)
        {
            alloc.active = false;
        }
    }
}
