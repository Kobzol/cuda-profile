import os

from conftest import offset_line, INPUT_FILENAME, metadata_file, kernel_file, run_file, param_all_formats


def check_debug_record(data, record, name, line):
    assert os.path.basename(record["file"]) == INPUT_FILENAME
    assert record["name"] == name
    assert record["line"] == offset_line(line, data)


def test_metadata_empty_debug(profile):
    data = profile("__global__ void kernel() {}", with_main=True)
    assert len(data) == 2
    assert metadata_file("kernel") in data
    assert data[metadata_file("kernel")]["locations"] == []


def test_metadata_debug_location(profile):
    data = profile("""
    __global__ void kernel(int* p) {
            int x = *p;
            *p = 5;
        }""", with_metadata=True, with_main=True)
    records = data["mappings"][metadata_file("kernel")]["locations"]

    assert len(records) == 2

    check_debug_record(data, records[0], "load", 3)
    check_debug_record(data, records[1], "store", 4)


def test_metadata_debug_index(profile):
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
    accesses = data[kernel_file("kernel")]["accesses"]
    assert accesses[0]["debugId"] == 0
    assert accesses[1]["debugId"] == 1


def test_metadata_debug_index_missing(profile):
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
    }""", debug=False)
    accesses = data[kernel_file("kernel")]["accesses"]
    assert accesses[0]["debugId"] == -1
    assert accesses[1]["debugId"] == -1


def test_metadata_type_index(profile):
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
    }""")

    types = data[metadata_file("kernel")]["typeMap"]
    assert len(types) > 0

    access = data[kernel_file("kernel")]["accesses"][0]
    assert access["typeIndex"] == types.index("i32")


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
