from conftest import kernel_file, param_all_formats


@param_all_formats
def test_global_allocation(profile, format):
    data = profile("""
    #include <cstdio>
    __global__ void kernel(int* p) {
        *p = 5;
    }
    int main() {
        int* dptr;
        cudaMalloc(&dptr, sizeof(int) * 10);
        printf("%p\\n", dptr);
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }
    """, format=format, with_metadata=True)

    allocations = data["mappings"][kernel_file("kernel", format=format)]["allocations"]

    assert len(allocations) == 1
    assert allocations[0]["active"]
    assert allocations[0]["elementSize"] == 4
    assert allocations[0]["size"] == 40
    assert allocations[0]["typeString"] == "i32"
    assert allocations[0]["space"] == 0
    assert allocations[0]["address"] == data["stdout"].strip()


@param_all_formats
def test_shared_allocation(profile, format):
    data = profile("""
    #include <cstdio>
    __global__ void kernel() {
        __shared__ int arr[10];
        printf("%p\\n", arr);
    }
    int main() {
        kernel<<<1, 1>>>();
        return 0;
    }
    """, format=format, with_metadata=True)

    allocations = data["mappings"][kernel_file("kernel", format=format)]["allocations"]

    assert len(allocations) == 1
    assert allocations[0]["active"]
    assert allocations[0]["elementSize"] == 4
    assert allocations[0]["size"] == 40
    assert "typeIndex" in allocations[0]
    assert allocations[0]["space"] == 1
    assert allocations[0]["address"] == data["stdout"].strip()
