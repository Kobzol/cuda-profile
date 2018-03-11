#include "Benchmark.h"

void vectorAdd();

int main(int argc, char** argv)
{
    Benchmark vectorAddBench("vectorAdd");
    vectorAddBench.measure([]() {
        vectorAdd();
    }, 10);
    vectorAddBench.print();

    return 0;
}
