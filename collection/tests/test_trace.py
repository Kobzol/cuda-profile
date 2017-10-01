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
