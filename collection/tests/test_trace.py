from conftest import kernel_file, param_all_formats


@param_all_formats
def test_access_address_match(profile, format):
    data = profile("""
    #include <cstdio>
    __global__ void kernel(int* p) {
        *p = 5;
    }
    int main() {
        int* dptr;
        cudaMalloc(&dptr, sizeof(int));
        printf("%p\\n", dptr);
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }
    """, format=format, with_metadata=True)
    assert data["stdout"].strip() == data["mappings"][kernel_file("kernel",
                                                                  format=format)]["accesses"][0]["address"]


@param_all_formats
def test_access_type_and_name(profile, format):
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
def test_access_time(profile, format):
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
