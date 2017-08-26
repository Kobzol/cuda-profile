from conftest import kernel_file


def test_shared_access(profile):
    data = profile("""
    __global__ void kernel() {
        __shared__ int arr[10];
        arr[threadIdx.x] = 1;
    }
    int main() {
        kernel<<<1, 1>>>();
        return 0;
    }
    """)
    assert data[kernel_file("kernel")]["accesses"][0]["event"]["space"] == "shared"


def test_constant_access(profile):
    data = profile("""
    __constant__ int arr[10];
    __global__ void kernel() {
        int x = arr[1];
    }
    int main() {
        kernel<<<1, 1>>>();
        return 0;
    }
    """)
    assert data[kernel_file("kernel")]["accesses"][0]["event"]["space"] == "constant"


def test_global_access(profile):
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
    """)
    assert data[kernel_file("kernel")]["accesses"][0]["event"]["space"] == "global"


def test_shared_constant_access(profile):
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
    """)

    assert data[kernel_file("kernel")]["accesses"][0]["event"]["space"] == "constant"
    assert data[kernel_file("kernel")]["accesses"][1]["event"]["space"] == "shared"
