#include "StringUtils.h"

std::string StringUtils::trimStart(std::string input, const std::string& trim)
{
    if (input.find(trim) == 0)
    {
        return input.substr(trim.length());
    }
    return input;
}
