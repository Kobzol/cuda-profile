#pragma once

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <glob.h>
#include <ios>
#include <fstream>

namespace cupr
{
    inline void checkCudaCall(cudaError_t code, const char* file, int line)
    {
        if (code != cudaSuccess)
        {
            fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        }
    }
    /**
     * Returns current UNIX timestamp in milliseconds.
     */
    inline int64_t getTimestamp()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
    }
    inline int createDirectory(const std::string& name)
    {
        return mkdir(name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    inline std::vector<std::string> glob(const std::string& pat)
    {
        glob_t glob_result{};
        glob(pat.c_str(), GLOB_TILDE, nullptr, &glob_result);
        std::vector<std::string> ret;
        for (int i = 0; i < glob_result.gl_pathc; i++)
        {
            ret.emplace_back(glob_result.gl_pathv[i]);
        }
        globfree(&glob_result);

        return ret;
    }
    inline void copyFile(const std::string& from, const std::string& to)
    {
        std::ifstream src(from, std::ios::binary);
        std::ofstream dst(to, std::ios::binary);
        dst << src.rdbuf();
    }
}

#define CHECK_CUDA_CALL(ans) { cupr::checkCudaCall((ans), __FILE__, __LINE__); }
#define __universal__ __device__ __host__

// CUDA device declarations for intellisense
__device__ unsigned int atomicInc(unsigned int* address, unsigned int val);
__device__ long long clock64();
