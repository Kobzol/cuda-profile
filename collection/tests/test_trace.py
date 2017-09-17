from conftest import kernel_file


def test_access_address_match(profile):
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
    """, with_metadata=True)
    assert data["stdout"].strip() == data["mappings"][kernel_file("kernel")]["accesses"][0]["event"]["address"]


def test_access_type_and_name(profile):
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
    }""")
    info = data[kernel_file("kernel")]
    assert info["type"] == "trace"
    assert info["kernel"] == "kernel"


def test_access_time(profile):
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
    }""")
    info = (data[kernel_file("kernel", 0)], data[kernel_file("kernel", 1)])
    assert info[0]["timestamp"] < info[1]["timestamp"]
