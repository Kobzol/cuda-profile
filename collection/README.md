## CUDA instrumentation pass
This repository contains code for a LLVM instrumentation pass. When run on CUDA code,
it will collect memory accesses from kernels and write them to files.

### Dependencies
To build the instrumentation pass, you will need LLVM and Clang (version >= 4.0). It is
assumed that you already have CUDA installed (version 8.0 or 9.0).

    $ apt-get install build-essential cmake llvm-4.0-dev clang-4.0

You can compress the outputted memory accesses and serialize them with either Cap'n Proto or
Protobuf instead of JSON (recommended).

    $ apt-get install libprotobuf-dev protobuf-compiler capnproto capnproto-dev zlib1g-dev

To run tests you also need py.test.

    $ apt-get install python python-pip python-pytest
    $ pip install pycapnp

### Build
    mkdir build
    cd build
    cmake ..
    make -j

### Usage
To instrument your CUDA program, you have to compile it with Clang and you have to
include the header file `CuprRuntime.h` from the `include` folder in every .cu file
that you want to be instrumented.

You have to pass the location of the instrumentation shared library and some other
flags to Clang. Example `CMakeLists.txt` files can be found in the `samples` folder.
You can also use the one-liner below (it expects that the environment variable
`CUPR_BUILD_DIR` contains an absolute path to the project build directory).

    $ clang++ -std=c++14 --cuda-gpu-arch=sm_30 \
        -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
        -I${CUPR_BUILD_DIR}/include \
        -L${CUPR_BUILD_DIR}/runtime \
        -Xclang -load -Xclang ${CUPR_BUILD_DIR}/instrument/libinstrument.so \
        -lruntime \
        -z muldefs \
        -lcudart \
        -xcuda \
        <source files> -o cuda

When you run the instrumented program, it will store the extracted memory
accesses to a `cupr-<unix-timestamp>` directory. You can then use the web dashboard
to load those files and visualise the memory accesses.

### Parameters
You can modify the behaviour of the instrumentation with the following parameters.
They are passed using environment variables, either during the compilation
(compilation context) or when you run the instrumented program (program context). For
bool parameters, use `1` for true value and `0` for false value.

| Name | Type | Default | Context | Description |
| ---- |:----:|:--------|:-------:|:-----------:|
| INSTRUMENT_LOCALS | bool | false | compiler | Instrument accesses to local variables |
| KERNEL_REGEX | string | | compiler | Regex to filter kernels for instrumentation |
| BUFFER_SIZE | number | 1048576 | program | GPU buffer size for stored accesses |
| PRETTIFY | bool | false | program | Prettify JSON output |
| COMPRESS | bool | false | program | Compress stored accesses |
| FORMAT | string | JSON | program | Serialization format (PROTOBUF, CAPNP or JSON) |
| HOST_MEMORY | bool | false | program | Use CPU memory for the stored accesses |

### Tests
    py.test tests

### Docker
You can use the provided Dockerfile to build an image with the profiler. To run the image,
you need to have Nvidia graphics card and install the Nvidia Docker runtime.

    $ apt-get install nvidia-docker2
    $ docker build -t cuda-profiler .
    $ docker run --runtime=nvidia cuda-profiler
