from conftest import param_all_formats, run_file, kernel_file


@param_all_formats
def test_general_emit_nothing(profile, format):
    data = profile("", with_main=True)
    assert run_file() in data
    assert len(data) == 1


@param_all_formats
def test_general_no_include(profile, format):
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
    """, format=format, add_include=False)
    assert run_file() in data
    assert len(data) == 1


def test_general_release_mode(profile):
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
    """, release=True)
    assert kernel_file("kernel") in data
