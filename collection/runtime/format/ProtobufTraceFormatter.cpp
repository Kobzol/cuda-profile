#include "ProtobufTraceFormatter.h"

#ifdef CUPR_USE_PROTOBUF
    #include "protobuf/generated/memory-access.pb.h"
    #include "protobuf/generated/kernel-trace.pb.h"
    #include <google/protobuf/io/gzip_stream.h>
    #include <google/protobuf/io/zero_copy_stream_impl.h>

    using namespace google::protobuf::io;
    using namespace cupr::proto;
#endif

void cupr::ProtobufTraceFormatter::formatTrace(std::ostream& os, const std::string& kernel, DeviceDimensions dimensions,
                                               const std::vector<cupr::AccessRecord>& accesses,
                                               const std::vector<cupr::AllocRecord>& allocations, double start,
                                               double end, bool prettify, bool compress)
{
#ifdef CUPR_USE_PROTOBUF
    KernelTrace trace;
    for (auto& access: accesses)
    {
        auto buffer = trace.add_accesses();
        buffer->set_address(this->hexPointer(access.address));
        buffer->set_size(static_cast<google::protobuf::int32>(access.size));
        buffer->set_warpid(access.warpId);
        buffer->set_debugid(access.debugIndex);
        buffer->set_kind(static_cast<google::protobuf::int32>(access.kind));
        buffer->set_space(static_cast<google::protobuf::int32>(access.addressSpace));
        buffer->set_typeindex(static_cast<google::protobuf::int32>(access.type));
        buffer->set_timestamp(std::to_string(access.timestamp));
        buffer->mutable_threadidx()->set_x(access.threadIdx.x);
        buffer->mutable_threadidx()->set_y(access.threadIdx.y);
        buffer->mutable_threadidx()->set_z(access.threadIdx.z);
        buffer->mutable_blockidx()->set_x(access.blockIdx.x);
        buffer->mutable_blockidx()->set_y(access.blockIdx.y);
        buffer->mutable_blockidx()->set_z(access.blockIdx.z);
    }

    for (auto& allocation: allocations)
    {
        auto buffer = trace.add_allocations();
        buffer->set_address(this->hexPointer(allocation.address));
        buffer->set_size(static_cast<google::protobuf::int32>(allocation.size));
        buffer->set_elementsize(static_cast<google::protobuf::int32>(allocation.elementSize));
        buffer->set_space(static_cast<google::protobuf::int32>(allocation.addressSpace));
        buffer->set_active(allocation.active);
        buffer->set_name(allocation.name == nullptr ? "" : allocation.name);
        buffer->set_location(allocation.location == nullptr ? "" : allocation.location);

        if (allocation.type == nullptr)
        {
            buffer->set_typeindex(static_cast<google::protobuf::int32>(allocation.typeIndex));
        }
        else buffer->set_typestring(allocation.type);
    }

    trace.set_kernel(kernel);
    trace.set_start(start);
    trace.set_end(end);
    trace.set_type("trace");
    trace.mutable_griddim()->set_x(dimensions.grid.x);
    trace.mutable_griddim()->set_y(dimensions.grid.y);
    trace.mutable_griddim()->set_z(dimensions.grid.z);
    trace.mutable_blockdim()->set_x(dimensions.block.x);
    trace.mutable_blockdim()->set_y(dimensions.block.y);
    trace.mutable_blockdim()->set_z(dimensions.block.z);
    trace.set_warpsize(dimensions.warpSize);

    if (compress)
    {
        OstreamOutputStream stream(&os);
        GzipOutputStream gs(&stream);
        trace.SerializeToZeroCopyStream(&gs);
    }
    else trace.SerializeToOstream(&os);
#endif
}

std::string cupr::ProtobufTraceFormatter::getSuffix()
{
    return "protobuf";
}
