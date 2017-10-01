#include "Formatter.h"

#ifdef CUPR_USE_PROTOBUF
    #include "protobuf/generated/memory-access.pb.h"
    #include "protobuf/generated/kernel-trace.pb.h"
#endif

using namespace cupr::proto;

picojson::value cupr::Formatter::jsonify(const cupr::AccessRecord& record)
{
    return picojson::value(picojson::object{
            {"threadIdx", picojson::value(picojson::object {
                    {"x", picojson::value((double) record.threadIdx.x)},
                    {"y", picojson::value((double) record.threadIdx.y)},
                    {"z", picojson::value((double) record.threadIdx.z)}
            })},
            {"blockIdx",  picojson::value(picojson::object {
                    {"x", picojson::value((double) record.blockIdx.x)},
                    {"y", picojson::value((double) record.blockIdx.y)},
                    {"z", picojson::value((double) record.blockIdx.z)}
            })},
            {"warpId",    picojson::value((double) record.warpId)},
            {"debugId",   picojson::value((double) record.debugIndex)},
            {"address",   picojson::value(this->hexPointer(record.address))},
            {"kind",      picojson::value((double) (record.kind == AccessType::Read ? 0 : 1))},
            {"size",      picojson::value((double) record.size)},
            {"space",     picojson::value((double) record.addressSpace)},
            {"typeIndex", picojson::value((double) record.type)},
            {"timestamp", picojson::value((double) record.timestamp)}
    });
}

picojson::value cupr::Formatter::jsonify(const cupr::AllocRecord& record)
{
    std::string typeKey = "typeString";
    picojson::value typeValue;
    if (record.type == nullptr)
    {
        typeKey = "typeIndex";
        typeValue = picojson::value((double) record.typeIndex);
    } else typeValue = picojson::value(record.type);

    return picojson::value(picojson::object {
            {"address",     picojson::value(this->hexPointer(record.address))},
            {"size",        picojson::value((double) record.size)},
            {"elementSize", picojson::value((double) record.elementSize)},
            {"space",       picojson::value((double) record.addressSpace)},
            {typeKey,       typeValue},
            {"active",      picojson::value(record.active)}
    });
}

void cupr::Formatter::outputKernelTraceJson(std::ostream& os,
                                            const std::string& kernel,
                                            const std::vector<cupr::AccessRecord>& accesses,
                                            const std::vector<cupr::AllocRecord>& allocations,
                                            double start,
                                            double end,
                                            bool prettify)
{
    auto value = picojson::value(picojson::object {
            {"type", picojson::value("trace")},
            {"kernel", picojson::value(kernel)},
            {"allocations",  this->jsonify(allocations)},
            {"accesses",   this->jsonify(accesses)},
            {"start", picojson::value(start)},
            {"end", picojson::value(end)}
    });

    os << value.serialize(prettify);
}

void cupr::Formatter::outputKernelTraceProtobuf(std::ostream& os,
                                                const std::string& kernel,
                                                const std::vector<cupr::AccessRecord>& accesses,
                                                const std::vector<cupr::AllocRecord>& allocations,
                                                double start,
                                                double end)
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
        buffer->set_timestamp(access.timestamp);
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
    trace.SerializeToOstream(&os);
#endif
}

void cupr::Formatter::outputProgramRun(std::fstream& os,
                                       int64_t timestampStart,
                                       int64_t timestampEnd)
{
    auto value = picojson::value(picojson::object {
            {"type", picojson::value("run")},
            {"start", picojson::value((double) timestampStart)},
            {"end", picojson::value((double) timestampEnd)}
    });

    os << value.serialize(true);
}

std::string cupr::Formatter::hexPointer(const void* ptr)
{
    std::ostringstream address;
    address << ptr;
    return address.str();
}
