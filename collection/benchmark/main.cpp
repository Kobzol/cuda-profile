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
void reduction();

#define BENCHMARK(name, count)\
    Benchmark name##Bench(#name);\
    name##Bench.measure([]() {\
        name();\
    }, count);\
    name##Bench.print()

int main(int argc, char** argv)
{
    BENCHMARK(vectorAdd, 10);
    BENCHMARK(simpleGL, 5);
    BENCHMARK(mandelbrot, 5);
    BENCHMARK(reduction, 5);

    return 0;
}
#endif
