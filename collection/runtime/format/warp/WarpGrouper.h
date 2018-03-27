#pragma once

#include <vector>

#include "../../tracedata/AccessRecord.h"
#include "../../DeviceDimensions.h"
#include "Warp.h"

namespace cupr {
    class WarpGrouper
    {
    public:
        std::vector<Warp> groupWarps(AccessRecord* records, size_t recordCount, const DeviceDimensions& dimensions);
    };
}
