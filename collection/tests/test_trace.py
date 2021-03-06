from conftest import kernel_file, param_all_formats


@param_all_formats
def test_trace_type_and_name(profile, format):
    data = profile("""
    __global__ void kernel(int* p) {
        int x = *p;
        *p = 5;
    }
    int main() {
        int* dptr;
        cudaMalloc(&dptr, sizeof(int));
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }""", format=format)
    info = data[kernel_file("kernel", format=format)]
    assert info["type"] == "trace"
    assert info["kernel"] == "kernel"


@param_all_formats
def test_trace_time(profile, format):
    data = profile("""
    #include <cstdio>
    __global__ void kernel(int* p) {
        *p = 5;
    }
    int main() {
        int* dptr;
        cudaMalloc(&dptr, sizeof(int));
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }
    """, format=format)
    assert data[kernel_file("kernel", format=format)]["end"] > data[kernel_file("kernel", format=format)]["start"]


@param_all_formats
def test_trace_multiple_invocations(profile, format):
    data = profile("""
    #include <cstdio>
    __global__ void kernel(int* p) {
        *p = 5;
    }
    int main() {
        int* dptr;
        cudaMalloc(&dptr, sizeof(int));
        kernel<<<1, 1>>>(dptr);
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }
    """, format=format)
    for i in xrange(2):
        assert kernel_file("kernel", i, format) in data


@param_all_formats
def test_trace_multiple_time(profile, format):
    data = profile("""
    __global__ void kernel(int* p) {
        int x = *p;
        *p = 5;
    }
    int main() {
        int* dptr;
        cudaMalloc(&dptr, sizeof(int));
        kernel<<<1, 1>>>(dptr);
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }""", format=format)
    info = (data[kernel_file("kernel", 0, format)], data[kernel_file("kernel", 1, format)])
    assert info[0]["start"] < info[1]["start"]


@param_all_formats
def test_trace_dimensions(profile, format):
    data = profile("""
    __global__ void kernel() {
        int x = threadIdx.x;
    }
    int main() {
        dim3 gridDim(3, 4, 5);
        dim3 blockDim(6, 7, 8);
        
        kernel<<<gridDim, blockDim>>>();
        return 0;
    }
    """, format=format)
    grid = data[kernel_file("kernel", format=format)]["gridDim"]
    assert grid["x"] == 3
    assert grid["y"] == 4
    assert grid["z"] == 5

    block = data[kernel_file("kernel", format=format)]["blockDim"]
    assert block["x"] == 6
    assert block["y"] == 7
    assert block["z"] == 8


@param_all_formats
def test_trace_warp_size(profile, format):
    data = profile("""
    __global__ void kernel() {
        int x = threadIdx.x;
    }
    int main() {
        kernel<<<1, 1>>>();
        return 0;
    }
    """, format=format)
    assert data[kernel_file("kernel", format=format)]["warpSize"] == 32


@param_all_formats
def test_trace_bank_size(profile, format):
    data = profile("""
    __global__ void kernel() {
        int x = threadIdx.x;
    }
    int main() {
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
        kernel<<<1, 1>>>();
        
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        kernel<<<1, 1>>>();
        return 0;
    }
    """, format=format)
    assert data[kernel_file("kernel", format=format)]["bankSize"] == 4
    assert data[kernel_file("kernel", index=1, format=format)]["bankSize"] == 8


@param_all_formats
def test_trace_warp_id(profile, format):
    data = profile("""
    __global__ void kernel(int* p) {
        *p = threadIdx.x;
    }
    int main() {
        int* dptr;
        cudaMalloc(&dptr, sizeof(int) * 64);
        kernel<<<1, 64>>>(dptr);
        cudaFree(dptr);
        return 0;
    }
    """, format=format)
    warps = data[kernel_file("kernel", format=format)]["warps"]
    for warp in warps:
        assert len(warp["accesses"]) == 32


@param_all_formats
def test_trace_thread_id(profile, format):
    data = profile("""
    __global__ void kernel(int* p) {
        *p = threadIdx.x;
    }
    int main() {
        int* dptr;
        cudaMalloc(&dptr, sizeof(int) * 64);
        kernel<<<1, 2>>>(dptr);
        cudaFree(dptr);
        return 0;
    }
    """, format=format)
    warp = data[kernel_file("kernel", format=format)]["warps"][0]
    assert warp["blockIdx"]["x"] == 0
    assert warp["blockIdx"]["y"] == 0
    assert warp["blockIdx"]["z"] == 0

    ids = ["{}.{}.{}".format(a["threadIdx"]["z"], a["threadIdx"]["y"], a["threadIdx"]["x"]) for a in warp["accesses"]]

    assert "0.0.0" in ids
    assert "0.0.1" in ids
