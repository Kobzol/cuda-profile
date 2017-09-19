#include "Emitter.h"

#include <fstream>
#include "Formatter.h"
#include "Parameters.h"

using namespace cupr;


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

void Emitter::emitKernelData(const std::string& kernelName, const std::vector<AccessRecord>& records,
                             const std::vector<AllocRecord>& allocations, float duration)
{
    std::cerr << "Emmitted " << records.size() << " accesses " << "in kernel " << kernelName << std::endl;

    std::string kernelFile = std::string(kernelName) + "-" + std::to_string(this->kernelCounter++);
    if (Parameters::isProtobufEnabled())
    {
        emitKernelDataProtobuf(kernelFile, kernelName, records, allocations, duration, getTimestamp());
    }
    else emitKernelDataJson(kernelFile, kernelName, records, allocations, duration, getTimestamp());
}
