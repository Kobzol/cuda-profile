from conftest import kernel_file, param_all_formats


@param_all_formats
def test_shared_access(profile, format):
    data = profile("""
    __global__ void kernel() {
        __shared__ int arr[10];
        arr[threadIdx.x] = 1;
    }
    int main() {
        kernel<<<1, 1>>>();
        return 0;
    }
    """, format=format)
    assert data[kernel_file("kernel", format=format)]["accesses"][0]["space"] == 1


@param_all_formats
def test_constant_access(profile, format):
    data = profile("""
    __constant__ int arr[10];
    __global__ void kernel() {
        int x = arr[1];
    }
    int main() {
        kernel<<<1, 1>>>();
        return 0;
    }
    """, format=format)
    assert data[kernel_file("kernel", format=format)]["accesses"][0]["space"] == 2


@param_all_formats
def test_global_access(profile, format):
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
    """, format=format)
    assert data[kernel_file("kernel", format=format)]["accesses"][0]["space"] == 0


@param_all_formats
def test_shared_constant_access(profile, format):
    data = profile("""
    __constant__ int constArr[10];
    __global__ void kernel() {
        __shared__ int arr[10];
        arr[threadIdx.x] = constArr[1];
    }
    int main() {
        kernel<<<1, 1>>>();
        return 0;
    }
    """, format=format)

    assert data[kernel_file("kernel", format=format)]["accesses"][0]["space"] == 2
    assert data[kernel_file("kernel", format=format)]["accesses"][1]["space"] == 1
