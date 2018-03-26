from conftest import param_all_formats, kernel_file


class AccessType:
    Read = 0
    Write = 1


@param_all_formats
def test_access_address_match(profile, format):
    data = profile("""
    #include <iostream>
    #include <iomanip>
    __global__ void kernel(int* p) {
        *p = 5;
    }
    int main() {
        int* dptr;
        cudaMalloc(&dptr, sizeof(int));

        size_t value = reinterpret_cast<size_t>(dptr);
        std::cout << std::uppercase << std::hex << std::setfill('0') << "0x" << std::setw(16) << value << std::endl;

        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }
    """, format=format, with_metadata=True)
    assert data["stdout"].strip() == data["mappings"][kernel_file("kernel",
                                                                  format=format)]["warps"][0]["accesses"][0]["address"]


@param_all_formats
def test_access_type(profile, format):
    data = profile("""
    #include <cstdio>
    __global__ void kernel(int* p) {
        *p = 5;
        int a = *p;
    }
    int main() {
        int* dptr;
        cudaMalloc(&dptr, sizeof(int));
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }
    """, format=format, with_metadata=True)
    warps = data["mappings"][kernel_file("kernel", format=format)]["warps"]

    assert warps[0]["kind"] == AccessType.Write
    assert warps[1]["kind"] == AccessType.Read


@param_all_formats
def test_access_value_int(profile, format):
    data = profile("""
    #include <cstdio>
    __global__ void kernel(int* p) {
        p[threadIdx.x] = *p;
    }
    int main() {
        int data = 1337;
        int* dptr;
        cudaMalloc(&dptr, sizeof(data));
        cudaMemcpy(dptr, &data, sizeof(data), cudaMemcpyHostToDevice);
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }
    """, format=format, with_metadata=True)
    warps = data["mappings"][kernel_file("kernel", format=format)]["warps"]

    assert warps[0]["accesses"][0]["value"] == "0x0000000000000539"
    assert warps[1]["accesses"][0]["value"] == "0x0000000000000539"


@param_all_formats
def test_access_value_float(profile, format):
    data = profile("""
    #include <cstdio>
    __global__ void kernel(float* p) {
        p[threadIdx.x] = *p;
    }
    int main() {
        float data = 1337;
        float* dptr;
        cudaMalloc(&dptr, sizeof(data));
        cudaMemcpy(dptr, &data, sizeof(data), cudaMemcpyHostToDevice);
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }
    """, format=format, with_metadata=True)
    warps = data["mappings"][kernel_file("kernel", format=format)]["warps"]

    assert warps[0]["accesses"][0]["value"] == "0x0000000000000539"
    assert warps[1]["accesses"][0]["value"] == "0x0000000000000539"


@param_all_formats
def test_access_value_bool(profile, format):
    data = profile("""
    #include <cstdio>
    __global__ void kernel(bool* p) {
        p[threadIdx.x] = *p;
    }
    int main() {
        bool data = true;
        bool* dptr;
        cudaMalloc(&dptr, sizeof(data));
        cudaMemcpy(dptr, &data, sizeof(data), cudaMemcpyHostToDevice);
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }
    """, format=format, with_metadata=True)
    warps = data["mappings"][kernel_file("kernel", format=format)]["warps"]

    assert warps[0]["accesses"][0]["value"] == "0x0000000000000001"
    assert warps[1]["accesses"][0]["value"] == "0x0000000000000001"


@param_all_formats
def test_access_value_ptr(profile, format):
    data = profile("""
    #include <iostream>
    #include <iomanip>
    __global__ void kernel(size_t* p) {
        p[threadIdx.x] = (size_t) p;
    }
    int main() {
        size_t data;
        size_t* dptr;
        cudaMalloc(&dptr, sizeof(data));
        cudaMemcpy(dptr, &data, sizeof(data), cudaMemcpyHostToDevice);
        kernel<<<1, 1>>>(dptr);
        
        size_t value = reinterpret_cast<size_t>(dptr);
        std::cout << std::uppercase << std::hex << std::setfill('0') << "0x" << std::setw(16) << value << std::endl;

        cudaFree(dptr);
        return 0;
    }
    """, format=format, with_metadata=True)
    warps = data["mappings"][kernel_file("kernel", format=format)]["warps"]

    assert data["stdout"].strip() == warps[0]["accesses"][0]["value"]


@param_all_formats
def test_access_complex0(profile, format):
    data = profile("""
    #include <cstdio>
    __global__ void kernel(float* a, float* b, float* c) {
        int tid = threadIdx.x;
        a[tid] = b[tid] + c[tid];
    }
    int main() {
        float* dptr;
        cudaMalloc(&dptr, sizeof(float));
        kernel<<<1, 1>>>(dptr, dptr, dptr);
        cudaFree(dptr);
        return 0;
    }
    """, format=format, with_metadata=True)
    warps = data["mappings"][kernel_file("kernel", format=format)]["warps"]

    assert warps[0]["kind"] == AccessType.Read
    assert warps[1]["kind"] == AccessType.Read
    assert warps[2]["kind"] == AccessType.Write

    assert warps[0]["size"] == 4
    assert warps[1]["size"] == 4
    assert warps[2]["size"] == 4


@param_all_formats
def test_access_local(profile, format):
    data = profile("""
    #include <cstdio>
    __global__ void kernel(float* p) {
        int a = 5;
        int b = a;
        *p = 5;
    }
    int main() {
        float* dptr;
        cudaMalloc(&dptr, sizeof(float));
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }
    """, format=format, with_metadata=True)
    accesses = data["mappings"][kernel_file("kernel", format=format)]["warps"][0]["accesses"]

    assert len(accesses) == 1
