from conftest import param_all_formats, kernel_file, run_file, requires_protobuf, metadata_file


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


@param_all_formats
def test_parameters_instrument_locals_disabled(profile, format):
    data = profile("""
    __global__ void kernel(int* p) {
        int a = 5;
        *p = a;
    }
    int main() {
        int* dptr;
        cudaMalloc(&dptr, sizeof(int));
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }""", format=format)

    accesses = data[kernel_file("kernel", format=format)]["accesses"]

    assert len(accesses) == 1


@param_all_formats
def test_parameters_instrument_locals_enable(profile, format):
    data = profile("""
    __global__ void kernel(int* p) {
        int a = 5;
        *p = a;
    }
    int main() {
        int* dptr;
        cudaMalloc(&dptr, sizeof(int));
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }""", format=format, instrument_locals=True)

    accesses = data[kernel_file("kernel", format=format)]["accesses"]

    assert len(accesses) == 4


@param_all_formats
def test_parameters_kernel_regex(profile, format):
    data = profile("""
    __global__ void addVectors(int* p) {
        int a = 5;
        *p = a;
    }
    __global__ void subtractVectors(int* p) {
        int a = 5;
        *p = a;
    }
    __global__ void generalKernel(int* p) {
        int a = 5;
        *p = a;
    }
    int main() {
        int* dptr;
        cudaMalloc(&dptr, sizeof(int));
        addVectors<<<1, 1>>>(dptr);
        subtractVectors<<<1, 1>>>(dptr);
        generalKernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }""", format=format, kernel_regex=".*a.*ors.*", with_metadata=True)

    mappings = data["mappings"]

    assert metadata_file("addVectors") in mappings
    assert metadata_file("subtractVectors") in mappings
    assert metadata_file("generalKernel") not in mappings

    assert len(mappings[kernel_file("addVectors", format=format)]["accesses"]) == 1
    assert len(mappings[kernel_file("subtractVectors", format=format)]["accesses"]) == 1
    assert kernel_file("generalKernel", format=format) not in mappings


@param_all_formats
def test_parameters_disable_output(profile, format):
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
    data = profile(code, disable_output=True)

    assert kernel_file("kernel", format=format) not in data
