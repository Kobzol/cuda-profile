#pragma once

#include <string>

class StringUtils
{
public:
    static std::string trimStart(std::string input, const std::string& trim);
    static std::string getFullPath(const std::string& path);
};
