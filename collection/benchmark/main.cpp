#ifdef COMPILE_ONLY
int main(int argc, char** argv)
{
    return 0;
}
#else

#include "Benchmark.h"

void vectorAdd();

#define BENCHMARK(name)\
    Benchmark name##Bench(#name);\
    name##Bench.measure([]() {\
        name();\
    }, 10);\
    name##Bench.print()

int main(int argc, char** argv)
{
    BENCHMARK(vectorAdd);

    return 0;
}
#endif
