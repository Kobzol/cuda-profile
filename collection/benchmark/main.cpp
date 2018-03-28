#ifdef COMPILE_ONLY
int main(int argc, char** argv)
{
    return 0;
}
#else

#include "Benchmark.h"

void vectorAdd();
void simpleGL();
void mandelbrot();

#define BENCHMARK(name, count)\
    Benchmark name##Bench(#name);\
    name##Bench.measure([]() {\
        name();\
    }, count);\
    name##Bench.print()

int main(int argc, char** argv)
{
    //BENCHMARK(vectorAdd, 5);
    BENCHMARK(simpleGL, 1);
    //BENCHMARK(mandelbrot, 1);

    return 0;
}
#endif
