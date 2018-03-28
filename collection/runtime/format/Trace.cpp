#include "Trace.h"

cupr::Trace::Trace(std::string kernelName, DeviceDimensions dimensions, cupr::AccessRecord* records, size_t recordCount,
                   const std::vector<cupr::AllocRecord>& allocations, float duration)
: kernelName(std::move(kernelName)), dimensions(dimensions), records(records), recordCount(recordCount),
  allocations(allocations), duration(duration)
{

}
