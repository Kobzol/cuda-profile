#include "Emitter.h"

#include "Formatter.h"
#include "../Parameters.h"

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

void Emitter::emitKernelTrace(const std::string& kernelName, dim3 dimensions[2],
                              const std::vector<AccessRecord>& records,
                              const std::vector<AllocRecord>& allocations, float duration)
{
    std::string kernelFile = this->getFilePath(std::string(kernelName) + "-" + std::to_string(this->kernelCounter++));
    auto timestamp = static_cast<double>(getTimestamp());
    double start = timestamp - static_cast<double>(duration);
    double end = timestamp;

    if (Parameters::isProtobufEnabled())
    {
        this->emitKernelTraceProtobuf(kernelFile, kernelName, dimensions, records, allocations, start, end);
    }
    else this->emitKernelTraceJson(kernelFile, kernelName, dimensions, records, allocations, start, end);
}

void Emitter::emitKernelTraceJson(const std::string& fileName,
                                  const std::string& kernel,
                                  dim3 dimensions[2],
                                  const std::vector<AccessRecord>& records,
                                  const std::vector<AllocRecord>& allocations,
                                  double start, double end)
{
    std::fstream kernelOutput(fileName + ".trace.json", std::fstream::out);

    Formatter formatter;
    formatter.outputKernelTraceJson(kernelOutput, kernel, dimensions, records, allocations, start, end,
                                    Parameters::isPrettifyEnabled());
    kernelOutput.flush();
}

void Emitter::emitKernelTraceProtobuf(const std::string& fileName,
                                      const std::string& kernel,
                                      dim3 dimensions[2],
                                      const std::vector<AccessRecord>& records,
                                      const std::vector<AllocRecord>& allocations,
                                      double start, double end)
{
    std::fstream kernelOutput(fileName + ".trace.protobuf", std::fstream::out);

    Formatter formatter;
    formatter.outputKernelTraceProtobuf(kernelOutput, kernel, dimensions, records, allocations, start, end);
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
