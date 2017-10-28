#include "Emitter.h"

#include "json/picojson.h"
#include "json/json_helper.h"

using namespace cupr;

Emitter::Emitter(std::unique_ptr<TraceFormatter> formatter, bool prettify, bool compress)
        : formatter(std::move(formatter)), prettify(prettify), compress(compress)
{
    this->directory = this->generateDirectoryName();
    createDirectory(this->directory);
    this->copyMetadataFiles();
}
void Emitter::emitProgramRun()
{
    std::ofstream runFile(this->getFilePath("run.json"));
    auto value = picojson::value(picojson::object {
            {"type", picojson::value("run")},
            {"start", picojson::value((double) this->timestampStart)},
            {"end", picojson::value((double) getTimestamp())},
            {"compress", picojson::value(this->compress)},
            {"traces", jsonify(this->traceFiles)}
    });

    runFile << value.serialize(true);
    runFile.flush();
}

void Emitter::emitKernelTrace(const std::string& kernelName, const DeviceDimensions& dimensions,
                              const std::vector<AccessRecord>& records,
                              const std::vector<AllocRecord>& allocations, float duration)
{
    if (this->kernelCount.find(kernelName) == this->kernelCount.end())
    {
        this->kernelCount.insert({kernelName, 0});
    }

    int count = this->kernelCount[kernelName]++;

    std::string filename = std::string(kernelName) +
            "-" +
            std::to_string(count) +
            ".trace." +
            this->formatter->getSuffix();
    std::string filepath = this->getFilePath(filename);
    auto timestamp = static_cast<double>(getTimestamp());
    double start = timestamp - static_cast<double>(duration);
    double end = timestamp;

    this->traceFiles.push_back(filename);

    std::ofstream kernelOutput(filepath);
    this->formatter->formatTrace(kernelOutput, kernelName, dimensions, records, allocations,
                                 start, end, this->prettify, this->compress);
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
    for (auto& metadata : glob("*.metadata.json"))
    {
        std::string fileName = metadata.substr(metadata.find_last_of('/') + 1);
        copyFile(metadata, this->getFilePath(fileName));
    }
}
