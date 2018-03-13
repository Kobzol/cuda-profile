import sys

import subprocess

import time

CUPR_SRC_DIR = "../"
CUPR_BUILD_DIR = "../cmake-build-release"
BENCH_SRC = "main.cpp"
INSTRUMENT = sys.argv[1] == '1' if len(sys.argv) > 1 else False
ITERATIONS = 10

# vectorAdd
SRC_FILES = ["samples/vectorAdd/vectorAdd.cu"]

LIBS = "-lcudart -lGL -lglut -lGLU"

times = []
for i in xrange(ITERATIONS):
    args = [
        "clang++",
        "-O0",
        "-g",
        "-std=c++14",
        "--cuda-gpu-arch=sm_30",
        "-I/usr/local/cuda/include",
        "-I/usr/local/cuda/samples/common/inc",
        "-L/usr/local/cuda/lib64",
        "-DCOMPILE_ONLY",
        "-lcudart", "-lGL", "-lglut", "-lGLU",
        "-xcuda"
    ]

    if INSTRUMENT:
        args += [
            "-I{}".format(CUPR_SRC_DIR),
            "-DPROFILE",
            "-Xclang", "-load", "-Xclang", "{}/instrument/libinstrument.so".format(CUPR_BUILD_DIR),
            "-z", "muldefs",
            "-L{}/runtime".format(CUPR_BUILD_DIR),
            "-lruntime"
        ]

    args += ["main.cpp"]
    args += SRC_FILES
    args += ["-o", "cuda"]

    start = time.time()
    process = subprocess.Popen(args,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    (out, err) = process.communicate()
    assert process.returncode == 0
    times.append(time.time() - start)

print(sum(times) / len(times))
