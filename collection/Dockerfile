FROM nvidia/cuda:8.0-devel

WORKDIR /cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    git \
    python python-pytest python-pip python-dev \
    llvm-4.0-dev clang-4.0 \
    libprotobuf-dev protobuf-compiler && \
    pip install wheel protobuf cython

RUN pip install setuptools --upgrade

RUN git clone https://github.com/capnproto/capnproto --depth=1 && \
    cd capnproto && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=1 ../ && \
    make -j install

RUN pip install pycapnp

RUN ln -s /usr/bin/clang++-4.0 /usr/bin/clang++

COPY . .

WORKDIR /cuda
RUN mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE .. && \
    make instrument runtime runtimetracker -j

WORKDIR /cuda
CMD BUILD_DIR=build py.test tests
