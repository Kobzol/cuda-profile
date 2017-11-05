#include "../runtime/tracking/Memtracker.h"

#include <dlfcn.h>
#include <driver_types.h>

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
    CU_PREFIX(malloc)(*devPtr, size, 1, "", "", "");
    return err;
}, void** devPtr, size_t size)
WRAP(cudaFree, cudaError_t, {
    CU_PREFIX(free)(devPtr);
    return cudaFree_orig(devPtr);
}, void* devPtr)
