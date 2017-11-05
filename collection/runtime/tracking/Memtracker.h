#pragma once

#include <cstddef>

#include "../Prefix.h"

extern "C" void CU_PREFIX(malloc)(void* address, size_t size, size_t elementSize, const char* type,
                                  const char* name, const char* location);
extern "C" void CU_PREFIX(free)(void* address);
