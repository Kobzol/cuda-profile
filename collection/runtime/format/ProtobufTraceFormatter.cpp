#include "ProtobufTraceFormatter.h"

#ifdef CUPR_USE_PROTOBUF
    #include <kernel-trace.pb.h>
    #include <google/protobuf/io/gzip_stream.h>
    #include <google/protobuf/io/zero_copy_stream_impl.h>

    using namespace google::protobuf::io;
    using namespace cupr::proto;
#endif

void cupr::ProtobufTraceFormatter::formatTrace(std::ostream& os, const std::string& kernel, DeviceDimensions dimensions,
                                               const std::vector<Warp>& warps,
                                               const std::vector<AllocRecord>& allocations, double start,
                                               double end, bool prettify, bool compress)
{
#ifdef CUPR_USE_PROTOBUF
    KernelTrace trace;
    for (auto& warp: warps)
    {
        auto buffer = trace.add_warps();
        buffer->set_size(static_cast<google::protobuf::int32>(warp.size));
        buffer->set_warpid(warp.id);
        buffer->set_debugid(warp.debugId);
        buffer->set_kind(static_cast<google::protobuf::int32>(warp.accessType));
        buffer->set_space(static_cast<google::protobuf::int32>(warp.space));
        buffer->set_typeindex(static_cast<google::protobuf::int32>(warp.typeId));
        buffer->set_timestamp(std::to_string(warp.timestamp));
        buffer->mutable_blockidx()->set_x(warp.blockIndex.x);
        buffer->mutable_blockidx()->set_y(warp.blockIndex.y);
        buffer->mutable_blockidx()->set_z(warp.blockIndex.z);

        for (auto& access: warp.accesses)
        {
            auto accessBuffer = buffer->add_accesses();
            accessBuffer->set_address(this->hexPointer(reinterpret_cast<const void*>(access.address)));
            accessBuffer->mutable_threadidx()->set_x(access.threadIndex.x);
            accessBuffer->mutable_threadidx()->set_y(access.threadIndex.y);
            accessBuffer->mutable_threadidx()->set_z(access.threadIndex.z);
            accessBuffer->set_value(this->hexPointer(reinterpret_cast<const void*>(access.value)));
        }
    }

    for (auto& allocation: allocations)
    {
        auto buffer = trace.add_allocations();
        buffer->set_address(this->hexPointer(allocation.address));
        buffer->set_size(static_cast<google::protobuf::int32>(allocation.size));
        buffer->set_elementsize(static_cast<google::protobuf::int32>(allocation.elementSize));
        buffer->set_space(static_cast<google::protobuf::int32>(allocation.addressSpace));
        buffer->set_active(allocation.active);
        buffer->set_location(allocation.location == nullptr ? "" : allocation.location);

        if (allocation.type == nullptr)
        {
            buffer->set_typeindex(static_cast<google::protobuf::int32>(allocation.typeIndex));
        }
        else buffer->set_typestring(allocation.type);

        if (allocation.name == nullptr)
        {
            buffer->set_nameindex(static_cast<google::protobuf::int32>(allocation.nameIndex));
        }
        else buffer->set_namestring(allocation.name);
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
    trace.set_banksize(dimensions.bankSize);

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
