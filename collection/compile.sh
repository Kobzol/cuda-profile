#!/usr/bin/env bash

LLVM_DIR=/home/kobzol/libraries/llvm-6.0.0-build
CLANG=clang++ #${LLVM_DIR}/bin/clang++
CUDA_DIR=/usr/local/cuda
SRC_FILES=${1-"../main.cpp ../kernel.cu"} # ../kernel2.cu
FORMAT=${2-"JSON"}
COMPRESS=${3-0}
RUNTIME_TRACKING=${4-0}
BUILD_DIR=${BUILD_DIR-"cmake-build-debug"}
INSTRUMENTED_KERNEL_BC="kernel-instrumented.bc"

pushd ${BUILD_DIR}
    # remove existing files
    # rm -rf *.bc *.ll *.o
    # rm -rf cupr-*

    # build pass
    make -j

    # ${CLANG} -I${CUDA_DIR}/samples/common/inc -O0 -c -emit-llvm -std=c++14 --cuda-gpu-arch=sm_30 ${SRC_FILES}

    # run pass and compile
    ${CLANG} -g -O0 -std=c++14 --cuda-gpu-arch=sm_30 \
            -I${CUDA_DIR}/include -L${CUDA_DIR}/lib64 \
            -I${CUDA_DIR}/samples/common/inc \
            -L./runtime \
            -Xclang -load -Xclang ./instrument/libinstrument.so \
            -z muldefs -lcudart -lruntime -xcuda \
            ${SRC_FILES} -o cuda

    if [ ${RUNTIME_TRACKING} == 1 ]; then
        export LD_PRELOAD="./runtimetracker/libruntimetracker.so"
    fi

    # run instrumented program
    LD_LIBRARY_PATH=./runtime FORMAT=${FORMAT} COMPRESS=${COMPRESS} ./cuda

    exit 0
popd

    # print compilation
    clang++ -### -std=c++14 --cuda-gpu-arch=sm_30 ${SRC_FILES} &> compilation.txt

    # emit LLVM bitcode
    clang++ -c -emit-llvm -std=c++14 --cuda-gpu-arch=sm_30 ${SRC_FILES}

    # run instrumentation
    opt -load instrument/libllvmCuda.so -cu *nvidia*.bc -o ${INSTRUMENTED_KERNEL_BC}

    # compile BC to object file
    llc -filetype=obj main.bc -o main.o

    llc -filetype=asm ${INSTRUMENTED_KERNEL_BC} -o kernel.s
    ptxas -m64 -O0 --gpu-name sm_30 --output-file kernel.ptxas kernel.s
    fatbinary --cuda -64 --create kernel.fatbin --image=profile=sm_30,file=kernel.ptxas --image=profile=compute_30,file=kernel.s
    clang++ -cc1 -triple x86_64-unknown-linux-gnu -aux-triple nvptx64-nvidia-cuda -emit-obj -mrelax-all -disable-free -disable-llvm-verifier -discard-value-names -main-file-name kernel.cu -mrelocation-model static -mthread-model posix -mdisable-fp-elim -fmath-errno -masm-verbose -mconstructor-aliases -munwind-tables -fuse-init-array -target-cpu x86-64 -dwarf-column-info -debugger-tuning=gdb -resource-dir /home/kobzol/libraries/llvm-4.0.0/bin/../lib/clang/4.0.0 -internal-isystem /home/kobzol/libraries/llvm-4.0.0/bin/../lib/clang/4.0.0/include/cuda_wrappers -internal-isystem /usr/local/cuda/include -include __clang_cuda_runtime_wrapper.h -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/5.4.0/../../../../include/c++/5.4.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/5.4.0/../../../../include/x86_64-linux-gnu/c++/5.4.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/5.4.0/../../../../include/x86_64-linux-gnu/c++/5.4.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/5.4.0/../../../../include/c++/5.4.0/backward -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/5.4.0/../../../../include/c++/5.4.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/5.4.0/../../../../include/x86_64-linux-gnu/c++/5.4.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/5.4.0/../../../../include/x86_64-linux-gnu/c++/5.4.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/5.4.0/../../../../include/c++/5.4.0/backward -internal-isystem /usr/local/include -internal-isystem /home/kobzol/libraries/llvm-4.0.0/bin/../lib/clang/4.0.0/include -internal-externc-isystem /usr/include/x86_64-linux-gnu -internal-externc-isystem /include -internal-externc-isystem /usr/include -internal-isystem /usr/local/include -internal-isystem /home/kobzol/libraries/llvm-4.0.0/bin/../lib/clang/4.0.0/include -internal-externc-isystem /usr/include/x86_64-linux-gnu -internal-externc-isystem /include -internal-externc-isystem /usr/include -std=c++14 -fdeprecated-macro -fdebug-compilation-dir /home/kobzol/projects/cuda/cmake-build-debug -ferror-limit 19 -fmessage-length 0 -fobjc-runtime=gcc -fcxx-exceptions -fexceptions -fdiagnostics-show-option -o kernel.o -x cuda ../kernel.cu -fcuda-include-gpubinary kernel.fatbin

    # link everything together
    clang++ main.o kernel.o -L${CUDA_DIR}/lib64 -lcudart_static -ldl -lrt -pthread -o cuda

    # run instrumented program
    ./cuda

    #/usr/bin/ld --hash-style=both --eh-frame-hdr -m elf_x86_64 -dynamic-linker /lib64/ld-linux-x86-64.so.2 -o a.out /usr/lib/gcc/x86_64-linux-gnu/5.4.0/../../../x86_64-linux-gnu/crt1.o /usr/lib/gcc/x86_64-linux-gnu/5.4.0/../../../x86_64-linux-gnu/crti.o /usr/lib/gcc/x86_64-linux-gnu/5.4.0/crtbegin.o -L/usr/lib/gcc/x86_64-linux-gnu/5.4.0 -L/usr/lib/gcc/x86_64-linux-gnu/5.4.0/../../../x86_64-linux-gnu -L/lib/x86_64-linux-gnu -L/lib/../lib64 -L/usr/lib/x86_64-linux-gnu -L/usr/lib/gcc/x86_64-linux-gnu/5.4.0/../../.. -L/home/kobzol/libraries/llvm-4.0.0/bin/../lib -L/lib -L/usr/lib /tmp/main-19efdd.o /tmp/kernel-c36455.o -lstdc++ -lm -lgcc_s -lgcc -lc -lgcc_s -lgcc /usr/lib/gcc/x86_64-linux-gnu/5.4.0/crtend.o /usr/lib/gcc/x86_64-linux-gnu/5.4.0/../../../x86_64-linux-gnu/crtn.o
popd
