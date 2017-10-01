from conftest import param_all_formats, kernel_file


class AccessType:
    Read = 0
    Write = 1


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
        printf("%p\\n", dptr);
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }
    """, format=format, with_metadata=True)
    accesses = data["mappings"][kernel_file("kernel", format=format)]["accesses"]

    assert accesses[0]["kind"] == AccessType.Write
    assert accesses[1]["kind"] == AccessType.Read
