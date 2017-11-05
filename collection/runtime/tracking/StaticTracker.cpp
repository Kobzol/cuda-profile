#include "../Prefix.h"
#include "Memtracker.h"

extern "C" void CU_PREFIX(malloc)(void* address, size_t size, size_t elementSize, const char* type,
                                  const char* name, const char* location)
{
    if (!CU_PREFIX(isRuntimeTrackingEnabled()))
    {
        cupr::memTracker.malloc(address, size, elementSize, type, name, location);
    }
}
extern "C" void CU_PREFIX(free)(void* address)
{
    if (!CU_PREFIX(isRuntimeTrackingEnabled()))
    {
        cupr::memTracker.free(address);
    }
}
