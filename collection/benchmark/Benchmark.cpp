#include "Benchmark.h"
#include "../runtime/CudaTimer.h"

#include <chrono>
#include <iostream>

TimeRecord::TimeRecord(double realTime, double cudaTime):
    realTime(realTime), cudaTime(cudaTime)
{

}

TimeRecord TimeRecord::operator+(const TimeRecord& other)
{
    return {this->realTime + other.realTime, this->cudaTime + other.cudaTime};
}

TimeRecord TimeRecord::operator/(double value)
{
    return {this->realTime / value, this->cudaTime  / value};
}

void TimeRecord::operator+=(const TimeRecord& other)
{
    this->realTime += other.realTime;
    this->cudaTime += other.cudaTime;
}

Benchmark::Benchmark(std::string name): name(std::move(name))
{

}

TimeRecord Benchmark::average() const
{
    TimeRecord sum;
    for (auto time: this->times)
    {
        sum += time;
    }
    return sum / this->times.size();
}

void Benchmark::measure(const std::function<void()>& body, size_t times)
{
    for (size_t i = 0; i < times; i++)
    {
        auto start = std::chrono::steady_clock::now();
        cupr::CudaTimer timer(false);

        timer.start();
        body();
        timer.stop_wait();

        this->times.emplace_back(
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count(),
                timer.get_time()
        );
    }
}

void Benchmark::print()
{
    auto avg = this->average();
    std::cout << this->name << ": " <<  avg.realTime << " CPU, " << avg.cudaTime << " GPU" << std::endl;
}
