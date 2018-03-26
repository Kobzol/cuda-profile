#include "RuntimeState.h"

#include "Parameters.h"
#include "format/JsonTraceFormatter.h"
#include "format/CapnpTraceFormatter.h"
#include "format/ProtobufTraceFormatter.h"

using namespace cupr;

static std::unique_ptr<TraceFormatter> createFormatter()
{
    auto format = Parameters::getFormat();
    if (format == Parameters::PROTOBUF_FORMAT)
    {
        return std::make_unique<ProtobufTraceFormatter>();
    }
    else if (format == Parameters::CAPNP_FORMAT)
    {
        return std::make_unique<CapnpTraceFormatter>();
    }
    else return std::make_unique<JsonTraceFormatter>();
}

RuntimeState::RuntimeState()
        : emitter(createFormatter(), Parameters::isPrettifyEnabled(), Parameters::isCompressionEnabled())
{

}

std::vector<AllocRecord>& RuntimeState::getAllocations()
{
    return this->allocations;
}

Emitter& RuntimeState::getEmitter()
{
    return this->emitter;
}
