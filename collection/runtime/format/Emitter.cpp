#include "Emitter.h"
#include "../Parameters.h"

#include "warp/WarpGrouper.h"

#define RAPIDJSON_HAS_STDSTRING 1
#include "json/rapidjson/ostreamwrapper.h"
#include "json/rapidjson/prettywriter.h"

using namespace cupr;
using namespace rapidjson;

Emitter::Emitter(std::unique_ptr<TraceFormatter> formatter, bool prettify, bool compress)
        : formatter(std::move(formatter)), prettify(prettify), compress(compress)
{
    this->directory = this->generateDirectoryName();
    createDirectory(this->directory);

    this->useThreadPool = Parameters::useThreadPool();

    if (this->useThreadPool)
    {
        this->pool = std::make_unique<FormatPool>();
    }
}
void Emitter::emitProgramRun()
{
    std::ofstream runFile(this->getFilePath("run.json"));

    OStreamWrapper stream(runFile);
    PrettyWriter<OStreamWrapper> writer(stream);

    writer.StartObject();

    writer.String("type");
    writer.String("run");
    writer.String("start");
    writer.Double(this->timestampStart);
    writer.String("end");
    writer.Double(getTimestamp());
    writer.String("compress");
    writer.Bool(this->compress);

    writer.String("traces");
    writer.StartArray();
    for (auto& file: this->traceFiles)
    {
        writer.String(file);
    }
    writer.EndArray();

    writer.EndObject();

    this->copyMetadataFiles();
}

void Emitter::emitKernelTrace(const std::string& kernelName, const DeviceDimensions& dimensions, AccessRecord* records,
                              size_t recordCount, const std::vector<AllocRecord>& allocations, float duration)
{
    if (Parameters::isOutputEnabled())
    {
        if (this->useThreadPool)
        {
            this->jobsProcessing++;

            this->pool->enqueue([this, kernelName, dimensions, records, recordCount, allocations, duration]() {
                this->emitKernelTraceJob(std::make_unique<Trace>(
                        kernelName, dimensions, records, recordCount, allocations, duration
                ));
                this->jobsProcessing--;
            });
        }
        else this->emitKernelTraceJob(std::make_unique<Trace>(
                    kernelName, dimensions, records, recordCount, allocations, duration
            ));
    }
    else delete[] records;
}

void Emitter::emitKernelTraceJob(std::unique_ptr<Trace> trace)
{
    int count = 0;

    {
        std::lock_guard<decltype(this->mutex)> guard(this->mutex);
        if (this->kernelCount.find(trace->kernelName) == this->kernelCount.end())
        {
            this->kernelCount.insert({trace->kernelName, 0});
        }

        count = this->kernelCount[trace->kernelName]++;
    }

    std::string filename = std::string(trace->kernelName) +
                           "-" +
                           std::to_string(count) +
                           ".trace." +
                           this->getTraceSuffix();
    std::string filepath = this->getFilePath(filename);
    auto timestamp = static_cast<double>(getTimestamp());
    double start = timestamp - static_cast<double>(trace->duration);
    double end = timestamp;

    {
        std::lock_guard<decltype(this->mutex)> guard(this->mutex);
        this->traceFiles.push_back(filename);
    }

    WarpGrouper grouper;
    auto warps = grouper.groupWarps(trace->records, trace->recordCount, trace->dimensions);

    auto mode = std::ios::out;
    if (this->formatter->isBinary())
    {
        mode |= std::ios::binary;
    }

    std::ofstream kernelOutput(filepath, mode);
    this->formatter->formatTrace(kernelOutput, trace->kernelName, trace->dimensions, warps, trace->allocations,
                                 start, end, this->prettify, this->compress);
    delete[] trace->records;
}

std::string Emitter::generateDirectoryName()
{
    return "cupr-" + std::to_string(getTimestamp());
}

std::string Emitter::getFilePath(const std::string& name)
{
    return this->directory + "/" + name;
}

void Emitter::copyMetadataFiles()
{
    for (auto& kernel: this->kernelCount)
    {
        auto meta = kernel.first + ".metadata.json";
        if (fileExists(meta))
        {
            copyFile(meta, this->getFilePath(meta));
        }
    }
}

std::string Emitter::getTraceSuffix()
{
    auto suffix = this->formatter->getSuffix();
    if (this->formatter->supportsCompression() && this->compress)
    {
        return "gzip." + suffix;
    }
    else return suffix;
}

void Emitter::waitForJobs()
{
    if (this->useThreadPool)
    {
        this->pool->stop();
        while (this->jobsProcessing > 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}
