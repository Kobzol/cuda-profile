import glob
import json
import subprocess
import tempfile
import os
import shutil
import pytest
from generated.kernel_invocation_pb2 import KernelInvocation
from google.protobuf import json_format

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
INSTRUMENT_LIB = "cmake-build-debug/instrument/libinstrument.so"
INPUT_FILENAME = "input.cu"


def create_test_dir():
    return tempfile.mkdtemp("cu")


def compile(root, lib, dir, code, format):
    inputname = INPUT_FILENAME
    outputname = "cuda"

    with open(os.path.join(dir, inputname), "w") as f:
        f.write(code)
        f.write("\n")

    args = ["clang++",
            "-g", "-O0",
            "-std=c++14",
            "--cuda-gpu-arch=sm_30",
            "-I/usr/local/cuda/include",
            "-L/usr/local/cuda/lib64",
            "-I{}".format(os.path.join(root, "runtime")),
            "-Xclang", "-load",
            "-Xclang", os.path.join(root, lib),
            "-lcudart", "-ldl", "-lrt", "-pthread",
            "-xcuda", inputname]

    if format == "protobuf":
        args += ["-lprotobuf"]
        args += glob.glob(os.path.join(root, "runtime/protobuf/generated/*.pb.cc"))
    args += ["-o", outputname]

    process = subprocess.Popen(args,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               cwd=dir)
    (out, err) = process.communicate()
    return (os.path.join(dir, outputname), process.returncode, out, err)


def run(dir, exe, env):
    runenv = os.environ.copy()
    runenv.update(env)

    process = subprocess.Popen([exe],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               cwd=dir,
                               env=runenv)
    (out, err) = process.communicate()

    mappings = {}

    for protobuf_file in glob.glob("{}/*.protobuf".format(dir)):
        with open(protobuf_file) as f:
            kernel = KernelInvocation()
            kernel.ParseFromString(f.read())
            mappings[os.path.basename(protobuf_file)] = json_format.MessageToDict(kernel,
                                                                                  preserving_proto_field_name=True)
    for json_file in glob.glob("{}/*.json".format(dir)):
        with open(json_file) as f:
            mappings[os.path.basename(json_file)] = json.load(f)

    return (mappings, process.returncode, out, err)


def compile_and_run(code,
                    add_include=True,
                    with_metadata=False,
                    with_main=False,
                    buffer_size=None,
                    format="json"):
    tmpdir = create_test_dir()

    env = {}
    if buffer_size is not None:
        env["CUPROFILE_BUFFER_SIZE"] = str(buffer_size)

    prelude = ""
    if add_include:
        if format == "protobuf":
            prelude += "#define CUPR_USE_PROTOBUF\n"
        prelude += "#include <Runtime.h>\n"

    if with_main:
        prelude += "int main() { return 0; }\n"

    code = prelude + code
    line_offset = len(prelude.splitlines())

    try:
        (exe, retcode, out, err) = compile(PROJECT_DIR, INSTRUMENT_LIB, tmpdir, code, format)

        if retcode != 0:
            raise Exception(str(retcode) + "\n" + out + "\n" + err)

        (mappings, retcode, out, err) = run(tmpdir, exe, env)
        if retcode != 0:
            raise Exception(retcode)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if with_metadata:
        return {
            "mappings": mappings,
            "stdout": out,
            "stderr": err,
            "line_offset": line_offset
        }
    else:
        return mappings


def offset_line(line, data):
    return line + data["line_offset"]


@pytest.fixture(scope="module")
def profile():
    return compile_and_run


def metadata_file(kernel="kernel"):
    return "{}.metadata.json".format(kernel)


def kernel_file(kernel="kernel", index=0, format="json"):
    return "{}-{}.trace.{}".format(kernel, index, format)


def param_all_formats(fn):
    @pytest.mark.parametrize("format", [
        "json",
        "protobuf"
    ])
    def inner_fn(profile, format):
        return fn(profile, format)
    return inner_fn
