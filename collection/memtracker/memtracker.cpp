#include "../runtime/RuntimeState.h"
#include "../runtime/Prefix.h"

#include <dlfcn.h>

extern "C" bool CU_PREFIX(isRuntimeTrackingEnabled)()
{
    return true;
}

#define WRAP(name, ret, body, ...)\
    using name##_t = ret (*)(__VA_ARGS__);\
    static name##_t name##_orig = nullptr;\
    extern "C" ret name(__VA_ARGS__) \
    {\
        if (name##_orig == nullptr)\
        {\
            name##_orig = (ret (*)(__VA_ARGS__)) dlsym(RTLD_NEXT, #name);\
        }\
        body\
    }\

WRAP(cudaMalloc, cudaError_t, {
    auto err = cudaMalloc_orig(devPtr, size);
    cupr::state.getAllocations().emplace_back(*devPtr, size, cupr::AddressSpace::Global);
    return err;
}, void** devPtr, size_t size)
