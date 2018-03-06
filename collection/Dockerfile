FROM nvidia/cuda:8.0-devel

WORKDIR /cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    python python-pytest python-pip \
    llvm-4.0-dev clang-4.0 \
    libprotobuf-dev protobuf-compiler && \
    pip install protobuf

RUN ln -s /usr/bin/clang++-4.0 /usr/bin/clang++

COPY . .
RUN mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE .. && \
    make -j

WORKDIR /cuda
CMD BUILD_DIR=build py.test tests
