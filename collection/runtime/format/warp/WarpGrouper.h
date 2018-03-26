#pragma once

#include <vector>

#include "../../tracedata/AccessRecord.h"
#include "../../DeviceDimensions.h"
#include "Warp.h"

namespace cupr {
    class WarpGrouper
    {
    public:
        std::vector<Warp> groupWarps(const std::vector<AccessRecord>& records, const DeviceDimensions& dimensions);
    };
}
