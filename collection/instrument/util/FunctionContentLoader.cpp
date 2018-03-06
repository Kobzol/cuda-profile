#include "FunctionContentLoader.h"

#include <fstream>
#include <sstream>

std::string FunctionContentLoader::loadFunction(const DebugInfo& info)
{
    std::ifstream fs(info.getFilename());
    std::stringstream buffer;
    buffer << fs.rdbuf();

    return buffer.str();
}
