import os

from conftest import offset_line, INPUT_FILENAME, metadata_file, kernel_file, run_file, param_all_formats, source_file


def check_debug_record(data, record, name, line):
    assert os.path.basename(record["file"]) == INPUT_FILENAME
    assert record["name"] == name
    assert record["line"] == offset_line(line, data)


def test_metadata_empty_debug(profile):
    data = profile("__global__ void kernel() {}", with_main=True)
    assert len(data) == 1
    assert metadata_file("kernel") not in data


def test_metadata_debug_location(profile):
    data = profile("""
    __global__ void kernel(int* p) {
        int x = *p;
        *p = 5;
    }
    int main() {
        int* dptr;
        cudaMalloc(&dptr, sizeof(int));
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }""", with_metadata=True)
    records = data["mappings"][metadata_file("kernel")]["locations"]

    assert len(records) == 2

    check_debug_record(data, records[0], "load", 3)
    check_debug_record(data, records[1], "store", 4)


@param_all_formats
def test_metadata_debug_index(profile, format):
    data = profile("""
    __global__ void kernel(int* p) {
        int x = *p;
        *p = 5;
    }
    int main() {
        int* dptr;
        cudaMalloc(&dptr, sizeof(int));
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }""", format=format)
    accesses = data[kernel_file("kernel", format=format)]["accesses"]
    assert accesses[0]["debugId"] == 0
    assert accesses[1]["debugId"] == 1


@param_all_formats
def test_metadata_debug_index_missing(profile, format):
    data = profile("""
    __global__ void kernel(int* p) {
        int x = *p;
        *p = 5;
    }
    int main() {
        int* dptr;
        cudaMalloc(&dptr, sizeof(int));
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }""", format=format, debug=False)
    accesses = data[kernel_file("kernel", format=format)]["accesses"]
    assert accesses[0]["debugId"] == -1
    assert accesses[1]["debugId"] == -1


@param_all_formats
def test_metadata_type_index(profile, format):
    data = profile("""
    __global__ void kernel(int* p) {
        int x = *p;
    }
    int main() {
        int* dptr;
        cudaMalloc(&dptr, sizeof(int));
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }""", format=format)

    types = data[metadata_file("kernel")]["typeMap"]
    assert len(types) > 0

    access = data[kernel_file("kernel", format=format)]["accesses"][0]
    assert access["typeIndex"] == types.index("i32")


@param_all_formats
def test_metadata_type_index_shared_buffer(profile, format):
    data = profile("""
    #include <cstdio>
    __global__ void kernel() {
        __shared__ float arr[10];
        arr[threadIdx.x] = threadIdx.x;
    }
    int main() {
        kernel<<<1, 1>>>();
        return 0;
    }
    """, format=format)

    types = data[metadata_file("kernel")]["typeMap"]
    assert len(types) > 0

    access = data[kernel_file("kernel", format=format)]["accesses"][0]
    assert access["typeIndex"] == types.index("float")


@param_all_formats
def test_metadata_type_index_shared_variable(profile, format):
    data = profile("""
    #include <cstdio>
    __global__ void kernel() {
        __shared__ float arr;
        arr = threadIdx.x;
    }
    int main() {
        kernel<<<1, 1>>>();
        return 0;
    }
    """, format=format)

    types = data[metadata_file("kernel")]["typeMap"]
    assert len(types) > 0

    access = data[kernel_file("kernel", format=format)]["accesses"][0]
    assert access["typeIndex"] == types.index("float")


@param_all_formats
def test_metadata_name_index_shared_buffer(profile, format):
    data = profile("""
    #include <cstdio>
    __global__ void kernel() {
        __shared__ float arr[10];
        arr[threadIdx.x] = threadIdx.x;
    }
    int main() {
        kernel<<<1, 1>>>();
        return 0;
    }
    """, format=format)

    names = data[metadata_file("kernel")]["nameMap"]
    assert len(names) > 0

    allocations = data[kernel_file("kernel", format=format)]["allocations"][0]
    assert allocations["nameIndex"] == names.index("arr")


def test_metadata_type_and_name(profile):
    data = profile("""
    __global__ void kernel(int* p) {
        int x = *p;
        *p = 5;
    }
    int main() {
        int* dptr;
        cudaMalloc(&dptr, sizeof(int));
        kernel<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }""")
    metadata = data[metadata_file("kernel")]
    assert metadata["type"] == "metadata"
    assert metadata["kernel"] == "kernel"


def test_metadata_run_time(profile):
    data = profile("__global__ void kernel() {}", with_main=True)
    run = data[run_file()]

    assert run["type"] == "run"
    assert run["end"] >= run["start"]


@param_all_formats
def test_metadata_run_traces(profile, format):
    data = profile("""
    __global__ void kernel(int* p) {
        *p = threadIdx.x;
    }
    __global__ void kernel2(int* p) {
        *p = threadIdx.x;
    }
    int main()
    {
        int* dptr;
        cudaMalloc(&dptr, sizeof(int));
        kernel<<<1, 1>>>(dptr);
        kernel<<<1, 1>>>(dptr);
        kernel2<<<1, 1>>>(dptr);
        cudaFree(dptr);
        return 0;
    }
    """, format=format)
    run = data[run_file()]

    traces = run["traces"]
    assert "kernel-0.trace." + format in traces
    assert "kernel-1.trace." + format in traces
    assert "kernel2-0.trace." + format in traces


def test_metadata_function_debug_info(profile):
    body = """// comment
        __global__ void kernel(int *p)
        {
            p[threadIdx.x] = threadIdx.x;
        }
        int main() {
            int* dptr;
            cudaMalloc(&dptr, sizeof(int));
            kernel<<<1, 1>>>(dptr);
            cudaFree(dptr);
            return 0;
        }
    """
    data = profile(body)
    metadata = data[metadata_file("kernel")]

    assert metadata["source"]["file"].endswith(source_file())
    assert metadata["source"]["line"] == 3
    assert metadata["source"]["content"].endswith(body)
