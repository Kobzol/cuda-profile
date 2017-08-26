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
    """, capture_io=True)
    assert data["stdout"].strip() == data["mappings"][kernel_file("kernel")]["accesses"][0]["event"]["address"]
