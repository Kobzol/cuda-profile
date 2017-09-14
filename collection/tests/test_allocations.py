from conftest import kernel_file


def test_global_allocation(profile):
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
    """, with_metadata=True)

    allocations = data["mappings"][kernel_file("kernel")]["memoryMap"]

    assert len(allocations) == 1
    assert allocations[0]["active"]
    assert allocations[0]["elementSize"] == 4
    assert allocations[0]["size"] == 40
    assert allocations[0]["type"] == "i32"
    assert allocations[0]["space"] == "global"
    assert allocations[0]["address"] == data["stdout"].strip()


def test_shared_allocation(profile):
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
    """, with_metadata=True)

    allocations = data["mappings"][kernel_file("kernel")]["memoryMap"]

    assert len(allocations) == 1
    assert allocations[0]["active"]
    assert allocations[0]["elementSize"] == 4
    assert allocations[0]["size"] == 40
    assert "typeIndex" in allocations[0]
    assert allocations[0]["space"] == "shared"
    assert allocations[0]["address"] == data["stdout"].strip()
