#include "Emitter.h"

#include <fstream>
#include "Formatter.h"
#include "Parameters.h"

using namespace cupr;

void Emitter::initialize()
{
    this->directory = this->generateDirectoryName();
    createDirectory(this->directory);
    this->copyMetadataFiles();
}
void Emitter::emitProgramRun()
{
    int64_t end = getTimestamp();

    std::fstream runFile(this->getFilePath("run.json"), std::fstream::out);
    Formatter formatter;
    formatter.outputProgramRun(runFile, this->timestampStart, end);
    runFile.flush();
}

void Emitter::emitKernelData(const std::string& kernelName, const std::vector<AccessRecord>& records,
                             const std::vector<AllocRecord>& allocations, float duration)
{
    std::string kernelFile = this->getFilePath(std::string(kernelName) + "-" + std::to_string(this->kernelCounter++));
    if (Parameters::isProtobufEnabled())
    {
        this->emitKernelDataProtobuf(kernelFile, kernelName, records, allocations, duration, getTimestamp());
    }
    else this->emitKernelDataJson(kernelFile, kernelName, records, allocations, duration, getTimestamp());
}

void Emitter::emitKernelDataJson(const std::string& fileName, const std::string& kernel,
                                 const std::vector<AccessRecord>& records, const std::vector<AllocRecord>& allocations,
                                 float duration, int64_t timestamp)
{
    std::fstream kernelOutput(fileName + ".trace.json", std::fstream::out);

    Formatter formatter;
    formatter.outputKernelRunJson(kernelOutput, kernel, records, allocations, duration, timestamp,
                                  Parameters::isPrettifyEnabled());
    kernelOutput.flush();
}

void Emitter::emitKernelDataProtobuf(const std::string& fileName, const std::string& kernel,
                                     const std::vector<AccessRecord>& records,
                                     const std::vector<AllocRecord>& allocations, float duration, int64_t timestamp)
{
    std::fstream kernelOutput(fileName + ".trace.protobuf", std::fstream::out);

    Formatter formatter;
    formatter.outputKernelRunProtobuf(kernelOutput, kernel, records, allocations, duration, timestamp);
    kernelOutput.flush();
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
