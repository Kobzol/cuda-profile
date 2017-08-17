#include "StringUtils.h"

std::string StringUtils::trimStart(std::string input, const std::string& trim)
{
    if (input.find(trim) == 0)
    {
        return input.substr(trim.length());
    }
    return input;
}

std::string StringUtils::getFullPath(const std::string& path)
{
    char* fullPath = realpath(path.c_str(), nullptr);
    std::string resolved(fullPath);
    free(fullPath);

    return resolved;
}
