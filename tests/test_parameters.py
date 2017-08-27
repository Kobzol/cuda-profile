def test_buffer_size_parameter(profile):
    code = """
    #include <cstdio>
    __global__ void kernel(int* p) {
        *p = 5;
        *p = 5;
        *p = 5;
    }
    int main() {
        int* dptr;
        cudaMalloc(&dptr, sizeof(int));
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }
    """

    data = profile(code, capture_io=True, buffer_size=2)
    assert data["stdout"].strip() == "DEVICE PROFILING OVERFLOW"

    data = profile(code, capture_io=True, buffer_size=20)
    assert data["stdout"].strip() == ""


def test_output_protobuf(profile):
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
    """, protobuf=True)

    assert "kernel-0.protobuf" in data
