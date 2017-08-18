import glob
import json
import subprocess
import tempfile
import os
import shutil
import pytest

PROJECT_DIR = "/home/kobzol/projects/cuda/cuda-profile"
INPUT_FILENAME = "input.cu"


def create_test_dir():
    return tempfile.mkdtemp("cu")


def compile(root, dir, code):
    inputname = INPUT_FILENAME
    outputname = "cuda"

    code = "#include <Runtime.h>\n" + code

    with open(os.path.join(dir, inputname), "w") as f:
        f.write(code)
        f.write("\n")

    process = subprocess.Popen(["clang++",
                                "-g",
                                "-O0",
                                "-std=c++14",
                                "--cuda-gpu-arch=sm_30",
                                "-I/usr/local/cuda/include",
                                "-L/usr/local/cuda/lib64",
                                "-I{}".format(os.path.join(root, "runtime")),
                                "-Xclang",
                                "-load",
                                "-Xclang",
                                os.path.join(root, "cmake-build-debug/instrument/libinstrument.so"),
                                "-lcudart",
                                "-ldl",
                                "-lrt",
                                "-pthread",
                                "-xcuda",
                                inputname,
                                "-o",
                                outputname],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               cwd=dir)
    (err, out) = process.communicate()
    return (os.path.join(dir, outputname), process.returncode, out, err)


def run(dir, exe):
    retcode = subprocess.Popen([exe], cwd=dir).wait()

    mappings = {}

    for json_file in glob.glob("{}/*.json".format(dir)):
        with open(json_file) as f:
            mappings[os.path.basename(json_file)] = json.load(f)

    return (retcode, mappings)


def compile_and_run(code):
    tmpdir = create_test_dir()

    try:
        (exe, retcode, out, err) = compile(PROJECT_DIR, tmpdir, code)

        if retcode != 0:
            raise Exception(str(retcode) + "\n" + out + "\n" + err)
        if err != "":
            raise Exception(str(retcode) + "\n" + out + "\n" + err)

        (retcode, mappings) = run(tmpdir, exe)
        if retcode != 0:
            raise Exception(retcode)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return mappings


def offset_line(line):
    return line + 1


@pytest.fixture(scope="module")
def profile():
    return compile_and_run


def with_main(code=""):
    return code + """
    int main() { return 0; }
    """


def debug_file(kernel="kernel"):
    return "debug-{}.json".format(kernel)


def kernel_file(kernel="kernel", index=0):
    return "{}-{}.json".format(kernel, index)
