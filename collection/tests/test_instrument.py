from conftest import kernel_file, param_all_formats, run_file


@param_all_formats
def test_no_include(profile, format):
    data = profile("""
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
    """, format=format, add_include=False)
    assert run_file() in data
    assert len(data) == 1


@param_all_formats
def test_kernel_time(profile, format):
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
    assert data[kernel_file("kernel", format=format)]["duration"] > 0


@param_all_formats
def test_multiple_invocations(profile, format):
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
