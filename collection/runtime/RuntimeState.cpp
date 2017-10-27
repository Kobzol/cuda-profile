#include "RuntimeState.h"
#include "Parameters.h"
#include "format/ProtobufTraceFormatter.h"
#include "format/JsonTraceFormatter.h"

using namespace cupr;

static std::unique_ptr<TraceFormatter> createFormatter()
{
    if (Parameters::isProtobufEnabled()) return std::make_unique<ProtobufTraceFormatter>();
    else return std::make_unique<JsonTraceFormatter>();
}

RuntimeState::RuntimeState()
{
    bool prettify = Parameters::isPrettifyEnabled();

    this->emitter.initialize(createFormatter(), prettify);
}

std::vector<AllocRecord>& RuntimeState::getAllocations()
{
    return this->allocations;
}

Emitter& RuntimeState::getEmitter()
{
    return this->emitter;
}
