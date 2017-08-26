import os

from conftest import with_main, offset_line, INPUT_FILENAME, metadata_file, kernel_file


def check_debug_record(record, name, line):
    assert os.path.basename(record["file"]) == INPUT_FILENAME
    assert record["name"] == name
    assert record["line"] == offset_line(line)


def test_emit_nothing(profile):
    data = profile(with_main())
    assert len(data) == 0


def test_emit_empty_debug(profile):
    data = profile(with_main("__global__ void kernel() {}"))
    assert len(data) == 1
    assert metadata_file("kernel") in data
    assert data[metadata_file("kernel")]["locations"] == []


def test_emit_debug_info(profile):
    data = profile(with_main("""
    __global__ void kernel(int* p) {
            int x = *p;
            *p = 5;
        }"""))
    records = data[metadata_file("kernel")]["locations"]

    assert len(records) == 2

    check_debug_record(records[0], "load", 3)
    check_debug_record(records[1], "store", 4)


def test_debug_index(profile):
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
    assert accesses[0]["debugId"] == 1
    assert accesses[1]["debugId"] == 2


def test_access_type_index(profile):
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

    access = data[kernel_file("kernel")]["accesses"][0]["event"]
    assert access["typeIndex"] == types.index("i32")
