import glob
import gzip
import json
import os
import pytest
import re
import shutil
import subprocess
import tempfile

try:
    from google.protobuf import json_format
    from generated.kernel_trace_pb2 import KernelTrace

    HAS_PROTOBUF = True
except ImportError:
    HAS_PROTOBUF = False

BUILD_DIR = os.environ.get("BUILD_DIR", "cmake-build-debug")
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
INSTRUMENT_LIB = "{0}/instrument/libinstrument.so".format(BUILD_DIR)
RUNTIME_LIB_DIR = "{0}/runtime".format(BUILD_DIR)
RUNTIME_TRACKING_LIB_DIR = "{0}/runtimetracker".format(BUILD_DIR)
COMPILER = os.environ.get("COMPILER", "clang++")
INCLUDE_DIR = "include"
INPUT_FILENAME = "input.cu"


def create_test_dir():
    return tempfile.mkdtemp("cu")


def compile(root, lib, dir, code, debug, instrument_locals):
    inputname = INPUT_FILENAME
    outputname = "cuda"

    env = os.environ.copy()
    if instrument_locals:
        env["CUPR_INSTRUMENT_LOCALS"] = "1"

    with open(os.path.join(dir, inputname), "w") as f:
        f.write(code)

    args = [COMPILER]

    if debug:
        args += ["-g", "-O0"]

    args += ["-std=c++14",
             "--cuda-gpu-arch=sm_30",
             "-I/usr/local/cuda/include",
             "-L/usr/local/cuda/lib64",
             "-L{}".format(os.path.join(root, RUNTIME_LIB_DIR)),
             "-I{}".format(os.path.join(root, INCLUDE_DIR)),
             "-Xclang", "-load",
             "-Xclang", os.path.join(root, lib),
             "-lcudart", "-ldl", "-lrt", "-lruntime",
             "-pthread",
             "-xcuda", inputname]

    args += ["-o", outputname]

    process = subprocess.Popen(args,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               cwd=dir,
                               env=env)
    (out, err) = process.communicate()
    return (os.path.join(dir, outputname), process.returncode, out, err)


def find_cupr_dir(dir):
    for (dirpath, _, _) in os.walk(dir):
        if re.search("cupr-\d+$", dirpath):
            return dirpath
    raise Exception("CUPR directory not found in {}".format(dir))


def run(root, dir, exe, env, compress):
    runenv = os.environ.copy()
    runenv.update(env)
    runenv["LD_LIBRARY_PATH"] = os.path.join(root, RUNTIME_LIB_DIR)

    process = subprocess.Popen([exe],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               cwd=dir,
                               env=runenv)
    (out, err) = process.communicate()

    mappings = {}
    cuprdir = find_cupr_dir(dir)

    if HAS_PROTOBUF:
        for protobuf_file in glob.glob("{}/*.protobuf".format(cuprdir)):
            with (gzip.open(protobuf_file) if compress else open(protobuf_file)) as f:
                kernel = KernelTrace()
                kernel.ParseFromString(f.read())
                mappings[os.path.basename(protobuf_file)] = json_format.MessageToDict(kernel,
                                                                                      preserving_proto_field_name=True)
    for json_file in glob.glob("{}/*.json".format(cuprdir)):
        with (gzip.open(json_file) if (compress and json_file.endswith(".gzip.json")) else open(json_file)) as f:
            mappings[os.path.basename(json_file)] = json.load(f)

    return (mappings, process.returncode, out, err)


def compile_and_run(code,
                    add_include=True,
                    with_metadata=False,
                    with_main=False,
                    buffer_size=None,
                    debug=True,
                    compress=False,
                    runtime_tracking=False,
                    instrument_locals=False,
                    format="json"):
    tmpdir = create_test_dir()

    env = {}
    if buffer_size is not None:
        env["CUPR_BUFFER_SIZE"] = str(buffer_size)
    if format == "protobuf":
        env["CUPR_PROTOBUF"] = "1"
    if compress:
        env["CUPR_COMPRESS"] = "1"
    if runtime_tracking:
        env["LD_PRELOAD"] = os.path.join(PROJECT_DIR, RUNTIME_TRACKING_LIB_DIR, "libruntimetracker.so")

    prelude = ""
    if add_include:
        prelude += "#include <CuprRuntime.h>\n"

    if with_main:
        prelude += "int main() { return 0; }\n"

    code = prelude + code
    line_offset = len(prelude.splitlines())

    try:
        (exe, retcode, out, err) = compile(PROJECT_DIR, INSTRUMENT_LIB, tmpdir, code, debug, instrument_locals)

        if retcode != 0:
            raise Exception(str(retcode) + "\n" + out + "\n" + err)

        (mappings, retcode, out, err) = run(PROJECT_DIR, tmpdir, exe, env, compress)
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


def kernel_file(kernel="kernel", index=0, format="json", compress=False):
    return "{}-{}.trace.{}{}".format(kernel, index, "gzip." if compress else "", format)


def run_file():
    return "run.json"


def source_file():
    return INPUT_FILENAME


def param_all_formats(fn):
    formats = ["json"]
    if HAS_PROTOBUF:
        formats.append("protobuf")

    @pytest.mark.parametrize("format", formats)
    def inner_fn(profile, format):
        return fn(profile, format)
    return inner_fn


requires_protobuf = pytest.mark.skipif(not HAS_PROTOBUF, reason="Protobuf is required")
