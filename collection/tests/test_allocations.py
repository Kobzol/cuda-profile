from conftest import kernel_file, param_all_formats


def validate_allocations(allocations):
    by_address = {}
    for alloc in allocations:
        if alloc["address"] in by_address and alloc["active"] and by_address[alloc["address"]]["active"]:
            raise Exception("Multiple active allocations")
        by_address[alloc["address"]] = alloc


@param_all_formats
def test_allocation_global(profile, format):
    data = profile("""
    #include <cstdio>
    __global__ void kernel(int* p) {
        *p = 5;
    }
    int main() {
        int* dptr;
        cudaMalloc((void**) &dptr, sizeof(int) * 10);
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
    assert allocations[0]["nameString"] == "dptr"
    assert allocations[0]["location"].endswith("input.cu:9")


@param_all_formats
def test_allocation_shared(profile, format):
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


def test_allocation_runtime_tracking_capture(profile):
    code = """
    __global__ void kernel(int* p) {
        *p = 5;
    }
    int main() {
        int* dptr;
        cudaMalloc((void**) &dptr, sizeof(int));
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }
    """

    data = profile(code, with_metadata=True, runtime_tracking=True)
    allocations = data["mappings"][kernel_file("kernel")]["allocations"]
    assert len(allocations) > 1

    data = profile(code, with_metadata=True)
    allocations = data["mappings"][kernel_file("kernel")]["allocations"]
    assert len(allocations) == 1


def test_parameters_runtime_tracking_overwrite(profile):
    code = """
    __global__ void kernel(int* p) {
        *p = 5;
    }
    int main() {
        int* dptr;
        cudaMalloc((void**) &dptr, sizeof(int) * 137); // try to find unique size
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }
    """

    data = profile(code, with_metadata=True)
    allocations = data["mappings"][kernel_file("kernel")]["allocations"]
    alloc = allocations[0]

    data = profile(code, with_metadata=True, runtime_tracking=True)
    allocations = data["mappings"][kernel_file("kernel")]["allocations"]
    validate_allocations(allocations)

    for record in allocations:
        if record["size"] == alloc["size"]:
            assert record["nameString"] == "dptr"
            return

    assert False  # allocation was not overwritten by static tracking
