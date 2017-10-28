#include "Emitter.h"

#include "picojson.h"

using namespace cupr;

Emitter::Emitter(std::unique_ptr<TraceFormatter> formatter, bool prettify)
        : formatter(std::move(formatter)), prettify(prettify)
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
            {"end", picojson::value((double) getTimestamp())}
    });

    runFile << value.serialize(true);
    runFile.flush();
}

void Emitter::emitKernelTrace(const std::string& kernelName, const DeviceDimensions& dimensions,
                              const std::vector<AccessRecord>& records,
                              const std::vector<AllocRecord>& allocations, float duration)
{
    std::string kernelFile = this->getFilePath(std::string(kernelName) + "-" + std::to_string(this->kernelCounter++));
    auto timestamp = static_cast<double>(getTimestamp());
    double start = timestamp - static_cast<double>(duration);
    double end = timestamp;

    std::ofstream kernelOutput(kernelFile + ".trace." + this->formatter->getSuffix());
    this->formatter->formatTrace(kernelOutput, kernelName, dimensions, records, allocations,
                                 start, end, this->prettify);
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
