from conftest import param_all_formats, kernel_file, run_file, requires_protobuf


def test_parameters_buffer_size(profile):
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

    data = profile(code, with_metadata=True, buffer_size=2)
    assert data["stdout"].strip() == "DEVICE PROFILING OVERFLOW"

    data = profile(code, with_metadata=True, buffer_size=20)
    assert data["stdout"].strip() == ""


@requires_protobuf
def test_parameters_protobuf(profile):
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
    """, format="protobuf")

    assert "kernel-0.trace.protobuf" in data


@param_all_formats
def test_parameters_compression_content(profile, format):
    code = """
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
    """

    uncompressed = profile(code, format=format)
    compressed = profile(code, format=format, compress=True)

    assert (uncompressed[kernel_file("kernel", format=format)]["kernel"] ==
            compressed[kernel_file("kernel", format=format, compress=True)]["kernel"])


@param_all_formats
def test_parameters_compression_run(profile, format):
    uncompressed = profile("", with_main=True, format=format)
    compressed = profile("", with_main=True, format=format, compress=True)

    assert not uncompressed[run_file()]["compress"]
    assert compressed[run_file()]["compress"]
