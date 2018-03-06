#pragma once

#include <string>
#include "DebugInfo.h"

class FunctionContentLoader
{
public:
    std::string loadFunction(const DebugInfo& info);
};
